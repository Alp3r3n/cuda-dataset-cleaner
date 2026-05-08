#include "kernel_launchers.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

__global__ void brightness_reduce_kernel(const uint8_t* gray, float* partial, int n) {
    extern __shared__ float sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = (i < n) ? (float)gray[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partial[blockIdx.x] = sdata[0];
}

extern "C"
float launch_brightness_kernel(const uint8_t* d_gray, int num_pixels, float* d_partial) {
    const int block_size = 256;
    int grid_size = (num_pixels + block_size - 1) / block_size;

    brightness_reduce_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
        d_gray, d_partial, num_pixels);
    cudaGetLastError();

    // The cudaMemcpy below is itself synchronizing on the default stream; the
    // explicit cudaDeviceSynchronize that used to live here was redundant.
    float* h_partial = new float[grid_size];
    cudaMemcpy(h_partial, d_partial, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < grid_size; i++) total += h_partial[i];
    delete[] h_partial;

    return total / (float)num_pixels;
}
