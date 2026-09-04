// gpu_stress.cu — GPU 算力压测（小矩阵乘，低显存占用）
// 用 cuBLAS 持续做 FP32 矩阵乘，把算力/功耗拉满，但只占约 300MB 显存。
// 适合叠加在推理负载上压测，不影响现役服务的显存分配。
//
// 采样与计算解耦：后台线程以固定间隔持续采样 nvidia-smi（温度/风扇/功耗/利用率）
// 写 CSV 日志；主线程只做 GEMM 不停机。这样温度曲线反映的是持续满载状态，
// 不受"跑完一批再采样"间隙的影响。
//
// 编译: nvcc -O3 -std=c++17 -o gpu_stress gpu_stress.cu -lcublas
// 运行: ./gpu_stress <Physical Slot> [矩阵边长N] [采样间隔秒]
//   槽位 = lspci -vv 的 Physical Slot 号（同 gpu_monitor.py 的 SLOT 列）；
//          该位置无卡则打印提示并退出。
//   N    = 矩阵边长，默认 4096（越大占显存越多）
//   间隔 = 采样间隔秒数，默认 1（温度曲线的时间粒度）

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <ctime>
#include <thread>
#include <chrono>
#include <atomic>
#include <cublas_v2.h>
#include <cuda_runtime.h>

static std::atomic<bool> g_stop{false};
static void on_sig(int s) { (void)s; g_stop = true; }

// 两线程共享的最新指标（采样线程写，主线程读用于屏幕显示）
static std::atomic<int>      m_util{-1}, m_temp{-1}, m_fan{-1}, m_mem{-1};
static std::atomic<long long> m_power_x100{-1};   // 功耗 x100（整数原子）
static std::atomic<double>   m_tflops{0.0};

#define CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "[错误] %s (line %d): %s\n", #call, __LINE__, cudaGetErrorString(err__)); \
        return 3; \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t cerr__ = (call); \
    if (cerr__ != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[错误] %s (line %d): cublas 状态=%d\n", #call, __LINE__, (int)cerr__); \
        return 3; \
    } \
} while(0)

// 读指定 GPU 的利用率/功耗/显存/风扇/温度（nvidia-smi）
static void read_gpu(int slot, int* util, double* poww, int* mem, int* fan, int* temp) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
             "nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,fan.speed,temperature.gpu "
             "--format=csv,noheader,nounits -i %d 2>/dev/null", slot);
    *util = *mem = *fan = *temp = -1;
    *poww = -1;
    FILE* fp = popen(cmd, "r");
    if (fp) {
        char line[512] = {0};
        if (fgets(line, sizeof(line), fp)) {
            // 分字段解析（fan.speed 可能是 "N/A"，逐项容错，不能让整个解析断掉）
            char* f0 = strtok(line, ",");
            char* f1 = strtok(NULL, ",");
            char* f2 = strtok(NULL, ",");
            char* f3 = strtok(NULL, ",");
            char* f4 = strtok(NULL, ",");
            if (f0) *util = atoi(f0);
            if (f1) *poww = atof(f1);
            if (f2) *mem  = atoi(f2);
            if (f3) {
                int v = atoi(f3);          // "N/A" → 0
                if (v > 0) *fan = v;
            }
            if (f4) *temp = atoi(f4);
        }
        fclose(fp);
    }
}

// 后台采样线程：固定间隔采样 nvidia-smi，更新共享指标并写 CSV
// nvidia_index = nvidia-smi 的 GPU index（用于 -i 查询）
// slot         = Physical Slot 号（用于 CSV 记录，用户视角）
static void sampler_thread(int nvidia_index, int slot, FILE* csv, long interval_ms) {
    while (!g_stop.load()) {
        int util = -1, mem = -1, fan = -1, temp = -1;
        double poww = -1;
        read_gpu(nvidia_index, &util, &poww, &mem, &fan, &temp);
        m_util.store(util);
        m_temp.store(temp);
        m_fan.store(fan);
        m_mem.store(mem);
        m_power_x100.store((long long)(poww * 100 + 0.5));
        if (csv) {
            char ts[64];
            time_t ct = time(NULL);
            struct tm* ctm = localtime(&ct);
            strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", ctm);
            fprintf(csv, "%s,%d,%d,%d,%d,%.0f,%d,%.2f\n",
                    ts, slot, util, temp, fan, poww, mem, m_tflops.load());
            fflush(csv);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "用法: %s <Physical Slot> [矩阵边长N] [采样间隔秒]\n"
            "  例:  %s 4\n"
            "  例:  %s 6 4096 1\n"
            "  槽位 = lspci -vv 的 Physical Slot 号（同 gpu_monitor.py 的 SLOT 列）\n"
            "        该位置无卡则打印可用槽位列表并退出\n"
            "  N    = 矩阵边长，默认 4096（越大占显存越多）\n"
            "  间隔 = 采样间隔秒数，默认 1（温度曲线的时间粒度）\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    int slot = atoi(argv[1]);
    int N    = (argc >= 3) ? atoi(argv[2]) : 4096;
    long interval_ms = (argc >= 4) ? atol(argv[3]) * 1000 : 1000;
    if (N <= 0) N = 4096;
    if (interval_ms < 100) interval_ms = 100;

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    int devCount = 0;
    CHECK(cudaGetDeviceCount(&devCount));

    // 通过 nvidia-smi 拿每张卡的 BDF（与 gpu_monitor.py 同源），
    // 再用 lspci -vv -s 解析 Physical Slot（lspci 的 "Physical Slot:" 行）
    int nvidia_index = -1;
    char avail_slots[128] = "";
    FILE* qfp = popen("nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null", "r");
    if (qfp) {
        char qline[256];
        while (fgets(qline, sizeof(qline), qfp)) {
            int idx;
            char busid[32] = {0};
            if (sscanf(qline, "%d,%31s", &idx, busid) != 2) continue;
            // "00000000:84:00.0" → "84:00.0"（跳过 domain）
            char* colon = strchr(busid, ':');
            char bdf_short[16];
            if (colon) snprintf(bdf_short, sizeof(bdf_short), "%s", colon + 1);
            else snprintf(bdf_short, sizeof(bdf_short), "%s", busid);

            char cmd[256], line[1024];
            int phys_slot = -1;
            snprintf(cmd, sizeof(cmd), "lspci -vv -s %s 2>/dev/null", bdf_short);
            FILE* fp = popen(cmd, "r");
            if (fp) {
                while (fgets(line, sizeof(line), fp)) {
                    char* ps = strstr(line, "Physical Slot:");
                    if (ps) { phys_slot = atoi(ps + 14); break; }
                }
                fclose(fp);
            }
            if (phys_slot < 0) phys_slot = 0;   // 拿不到 Physical Slot 回退 0（同 gpu_monitor.py）

            if (avail_slots[0]) strcat(avail_slots, ", ");
            char buf[16];
            snprintf(buf, sizeof(buf), "%d", phys_slot);
            strcat(avail_slots, buf);

            if (phys_slot == slot) nvidia_index = idx;
        }
        fclose(qfp);
    }

    if (nvidia_index < 0) {
        fprintf(stderr,
            "提示: Physical Slot %d 不存在。本机可用槽位: %s。退出。\n",
            slot, avail_slots);
        return 2;
    }

    cudaDeviceProp prop;
    CHECK(cudaSetDevice(nvidia_index));
    CHECK(cudaGetDeviceProperties(&prop, nvidia_index));

    double mb = (double)N * N * 4 / 1048576.0;   // 单矩阵 MiB
    printf("压测目标 : Physical Slot %d (nvidia-smi GPU %d) — %s（总显存 %.1f GB）\n",
           slot, nvidia_index, prop.name, prop.totalGlobalMem / 1073741824.0);
    printf("矩阵规模 : %d x %d x %d (FP32)，单矩阵 %.1f MiB\n", N, N, N, mb);
    printf("显存占用 : 约 %.0f MiB（3 块矩阵，停止后立即释放）\n", mb * 3);
    printf("采样间隔 : %ld ms（后台线程，独立于计算）\n", interval_ms);
    printf("开始压测… 按 Ctrl+C 停止\n");
    fflush(stdout);

    size_t bytes = (size_t)N * N * sizeof(float);
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));
    CHECK(cudaMemset(dA, 0x3c, bytes));   // 非零填充（压测不关心结果正确性）
    CHECK(cudaMemset(dB, 0x3c, bytes));
    CHECK(cudaMemset(dC, 0x00, bytes));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f, beta = 0.0f;
    // 预热（触发 cuBLAS 算法选择 / 首块分配）
    for (int i = 0; i < 3; i++)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          N, N, N, &alpha, dA, N, dB, N, &beta, dC, N));
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

    // CSV 日志：由采样线程按固定间隔写入，带时间戳，用于后期分析温度/功耗曲线
    time_t now = time(NULL);
    struct tm* lt = localtime(&now);
    char csvPath[512];
    snprintf(csvPath, sizeof(csvPath), "/data/gpu_stress/stress_gpu%d_%04d%02d%02d_%02d%02d%02d.csv",
             slot, lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
             lt->tm_hour, lt->tm_min, lt->tm_sec);
    FILE* csv = fopen(csvPath, "w");
    if (csv) {
        fprintf(csv, "timestamp,slot,util_pct,temp_c,fan_pct,power_w,mem_mib,tflops\n");
        printf("日志文件 : %s\n", csvPath);
    } else {
        fprintf(stderr, "[警告] 无法写入日志文件 %s，仅屏幕输出\n", csvPath);
    }
    fflush(stdout);

    // 启动后台采样线程
    std::thread sampler(sampler_thread, nvidia_index, slot, csv, interval_ms);

    // 主线程：持续 GEMM 不停机，定期同步测吞吐
    auto wall0 = std::chrono::steady_clock::now();
    long totalIter = 0;
    while (!g_stop) {
        CHECK(cudaEventRecord(ev0));
        long count = 0;
        for (long i = 0; i < 200 && !g_stop; i++) {
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, N, N, &alpha, dA, N, dB, N, &beta, dC, N));
            count++;
        }
        CHECK(cudaEventRecord(ev1));
        CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        totalIter += count;
        double tflops = (2.0 * N * N * N * count) / (ms / 1000.0) / 1e12;
        m_tflops.store((double)tflops);

        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
        printf("  [GPU %d] 温度 %3dC  风扇 %3d%%  利用率 %3d%%  功耗 %4.0f W  显存 %5d MiB  |  %6.2f TFLOPS  |  %.0f s\n",
               slot, m_temp.load(), m_fan.load(), m_util.load(),
               m_power_x100.load() / 100.0, m_mem.load(), tflops, el);
        fflush(stdout);
    }

    g_stop = true;
    sampler.join();

    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    double avg = (el > 0) ? (2.0 * N * N * N * totalIter) / (el * 1e12) : 0.0;
    printf("已停止。累计 %ld 次 GEMM，运行 %.0f 秒，平均 %.2f TFLOPS。显存已释放。\n",
           totalIter, el, avg);
    if (csv) {
        fclose(csv);
        printf("日志已保存: %s（%ld ms 一行，可直接导入绘图分析温度/功耗曲线）\n",
               csvPath, interval_ms);
    }

    CHECK(cudaEventDestroy(ev0));
    CHECK(cudaEventDestroy(ev1));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    CHECK(cudaDeviceReset());
    return 0;
}
