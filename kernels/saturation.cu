#include "kernel_launchers.hpp"
#include <cuda_runtime.h>
#include <cstdint>

__global__ void saturation_count_kernel(const uint8_t* gray, int* partial,
                                        uint8_t threshold, int n) {
    extern __shared__ int sdata_int[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata_int[threadIdx.x] = (i < n && gray[i] > threshold) ? 1 : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata_int[threadIdx.x] += sdata_int[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partial[blockIdx.x] = sdata_int[0];
}

extern "C"
float launch_saturation_kernel(const uint8_t* d_gray, int num_pixels,
                               float threshold, int* d_partial) {
    if (num_pixels <= 0) return 0.0f;

    const int block_size = 256;
    int grid_size = (num_pixels + block_size - 1) / block_size;

    uint8_t thresh_u8 = (threshold > 255.0f) ? 255
                      : (threshold < 0.0f)   ? 0
                      : (uint8_t)threshold;

    saturation_count_kernel<<<grid_size, block_size, block_size * sizeof(int)>>>(
        d_gray, d_partial, thresh_u8, num_pixels);
    cudaGetLastError();

    int* h_partial = new int[grid_size];
    cudaMemcpy(h_partial, d_partial, grid_size * sizeof(int), cudaMemcpyDeviceToHost);

    long long total = 0;
    for (int i = 0; i < grid_size; i++) total += h_partial[i];
    delete[] h_partial;

    return (float)total / (float)num_pixels;
}
