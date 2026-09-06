# GPU 算力压测工具（小矩阵乘，低显存占用 + 温度/功耗曲线记录）

用 cuBLAS 持续做 FP32 矩阵乘，把目标 GPU 的**算力/功耗拉满**，但只用约 **300MB 显存**（3 块小矩阵），跑完或停止后**立即释放**，不挤占推理服务的大块显存分配。适合**叠加在现役推理负载上**做算力/功耗/温控压测。

同时**后台采样线程**以固定间隔记录 nvidia-smi 的温度/风扇/功耗/利用率到 CSV，用于后期分析**温度曲线**。

## 文件

- `gpu_stress.cu` — CUDA 源码
- `gpu_stress` — 编译好的可执行（nvcc -O3，CUDA 13.3）
- `stress_gpu<槽位>_<时间戳>.csv` — 运行后自动生成的采样日志

## 重新编译（改了 .cu 之后）

```bash
cd /data/gpu_stress
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -o gpu_stress gpu_stress.cu -lcublas
```

## 运行

```bash
./gpu_stress <Physical Slot> [矩阵边长N] [采样间隔秒]
```

- **Physical Slot** = `lspci -vv` 的 PCIe 物理槽位号（同 gpu_monitor.py 的 SLOT 列）。脚本启动时经 nvidia-smi BDF → lspci 自动映射到对应 GPU。**该位置没有 GPU 时，打印可用槽位列表并退出**（退出码 2）。
- **N** = 矩阵边长，默认 4096（4096³ 单矩阵约 64MB，共约 192MB）。N 越大算力越高、占显存越多。
- **采样间隔秒** = 温度/功耗曲线的时间粒度，默认 1 秒（最小 0.1 秒）。

查本机可用槽位：`lspci -vv | grep -B1 "Physical Slot"`，或看 gpu_monitor 的 SLOT 列。

示例：

```bash
./gpu_stress 4              # 压测 slot 4 的 GPU，默认 4096、1 秒采样
./gpu_stress 6 8192 0.5     # 压测 slot 6 的 GPU，更大矩阵、0.5 秒采样
```

按 `Ctrl+C`（SIGINT/SIGTERM）停止，显存随即释放。

## 输出

**屏幕**（主线程每 2 秒左右一行）：

```
压测目标 : Physical Slot 4 (nvidia-smi GPU 1) — NVIDIA GeForce RTX 3090（总显存 23.6 GB）
硬件标识 : UUID GPU-c128393b-548d-13e9-1fac-7b0824c410da
矩阵规模 : 4096 x 4096 x 4096 (FP32)，单矩阵 64.0 MiB
显存占用 : 约 192 MiB（3 块矩阵，停止后立即释放）
采样间隔 : 1000 ms（后台线程，独立于计算）
日志文件 : /data/gpu_stress/stress_gpu4_20260906_103656.log
开始压测… 按 Ctrl+C 停止
  [GPU 4] 温度  79C  风扇  96%  利用率 100%  功耗  299 W  显存 22345 MiB  降频:否  |   24.27 TFLOPS  |  2 s
```

- **硬件标识** = 该卡的 GPU UUID（`nvidia-smi` 的唯一硬件标识），启动时打印一次，便于把日志对应到具体物理卡。
- **降频** = 是否发生热降频，口径同 `/root/gpu_monitor.py`：`clocks_throttle_reasons.active & 0x60`（SW Thermal 0x20 | HW Thermal 0x40 任一激活即为"是"）；读不到显示 "?"。

**日志文件**（内容 = 终端输出内容，每行前加时间戳，用于后期分析）：

```
[2026-09-06 10:36:56] 压测目标 : Physical Slot 4 (nvidia-smi GPU 1) — NVIDIA GeForce RTX 3090（总显存 23.6 GB）
[2026-09-06 10:36:56] 硬件标识 : UUID GPU-c128393b-548d-13e9-1fac-7b0824c410da
[2026-09-06 10:36:57]   [GPU 4] 温度  78C  风扇  96%  利用率 100%  功耗  299 W  显存 22345 MiB  降频:否  |   24.38 TFLOPS  |  1 s
[2026-09-06 10:36:58]   [GPU 4] 温度  79C  风扇  96%  利用率 100%  功耗  299 W  显存 22345 MiB  降频:否  |   24.27 TFLOPS  |  2 s
[2026-09-06 10:37:00] 已停止。累计 600 次 GEMM，运行 4 秒，平均 19.64 TFLOPS。显存已释放。
```

路径：`/data/gpu_stress/stress_gpu<槽位>_<日期>_<时间>.log`，每行写完立即 flush（中途停止也保留已有数据）。

## 设计要点：采样与计算解耦

后台采样线程以**固定间隔**独立调用 nvidia-smi，主线程只做 GEMM 不停机。这样：
- 温度曲线反映**持续满载**状态，每秒一个采样点，与矩阵大小、GEMM 进度无关
- 不会出现"跑完一批才采样"导致采样点落在 GPU 空闲间隙、温度/功耗偏低的问题
- 采样粒度固定且可配（第三个参数）

## 重要：占用的是「算力」，不是「显存」

- 显存只多占约 300MB，**不影响推理服务的显存分配**（这是"不占显存"的含义）。
- 但**算力是共享的**：压测某张卡时，**那张卡上正在跑的推理会明显变慢**（本工作站是双卡 TP 部署，GPU 0 和 GPU 1 都在跑推理）。
- 若只想测温度/功耗上限且不想影响推理，建议短暂压测或挑空闲时段；长时压测会拖慢现役服务。

## nvidia-smi 字段名（踩过的坑）

- 温度要用 **`temperature.gpu`**（核心 GPU 温度），写成 `temperature` 会报错导致整行查询失败。
- 还有 `temperature.memory`（显存温度）、`temperature.gpu.tlimit`（T.Limit 温度）。
- `fan.speed` 在被动散热卡上可能返回 `N/A`，代码已逐项容错（读不到记 -1，不连累其他字段）。

## 本机 GPU 基线（2026-09-04，供对照）

- GPU 0 / GPU 1：RTX 3090，各 24576 MiB 显存，双卡被 sglang 推理服务占用（各约 21880 MiB）
- 压测 GPU 1 时：显存 21891→22357 MiB（+460MB），利用率 100%，功耗约 270-292 W，温度约 89C，风扇 100%，约 10-14 TFLOPS（FP32 GEMM）
