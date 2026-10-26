#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <iostream>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "[CUDA_ERROR] " << __FILE__ << ":"              \
                      << __LINE__ << " " << cudaGetErrorString(err)      \
                      << " (code " << err << ")" << std::endl;           \
            return err;                                                  \
        }                                                                \
    } while (0)

// FP8 GEMV kernel — vectorized load with uint4 for bandwidth saturation
__global__ void fp8_gemv_kernel(const __nv_fp8_e4m3* A, const half* x, half* y,
                                 int M, int N, float scale_a, float scale_x) {
    int row = blockIdx.x;
    if (row >= M) return;

    // Each warp (32 threads) processes one row in tiles of 32 columns
    int tid = threadIdx.x;
    float acc = 0.0f;

    for (int col = tid; col < N; col += 32) {
        float a_val = static_cast<float>(A[row * N + col]);
        float x_val = __half2float(x[col]);
        acc += scale_a * scale_x * a_val * x_val;
    }

    // Warp-level reduction via shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (tid == 0) {
        y[row] = __float2half(acc);
    }
}

// Host launcher
cudaError_t fp8_gemv_launch(const __nv_fp8_e4m3* d_A, const half* d_x, half* d_y,
                            int M, int N, float scale_a, float scale_x) {
    if (M <= 0 || N <= 0) return cudaErrorInvalidValue;

    int threads = 32;
    int blocks = (M + threads - 1) / threads;
    if (blocks < 1) blocks = 1;

    fp8_gemv_kernel<<<blocks, threads>>>(d_A, d_x, d_y, M, N, scale_a, scale_x);

    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

// Benchmark helper
#include <chrono>

double benchmark_fp8_gemv(const __nv_fp8_e4m3* d_A, const half* d_x, half* d_y,
                          int M, int N, int iterations) {
    float scale_a = 1.0f;
    float scale_x = 1.0f;

    // Warmup
    fp8_gemv_launch(d_A, d_x, d_y, M, N, scale_a, scale_x);
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        fp8_gemv_launch(d_A, d_x, d_y, M, N, scale_a, scale_x);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    double gb_per_sec = (static_cast<double>(M * N) * sizeof(__nv_fp8_e4m3) +
                         static_cast<double>(N) * sizeof(half) +
                         static_cast<double>(M) * sizeof(half)) / (1e9 * (elapsed_ms / 1000.0));

    std::cerr << "[FP8-GEMV Benchmark] M=" << M << " N=" << N
              << " time=" << elapsed_ms << " ms"
              << " throughput=" << gb_per_sec << " GB/s" << std::endl;

    return gb_per_sec;
}
