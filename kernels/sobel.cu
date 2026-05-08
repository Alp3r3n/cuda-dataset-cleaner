#include "kernel_launchers.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

__global__ void sobel_edge_kernel(const uint8_t* gray, float* partial,
                                  int width, int height, int interior_pixels) {
    extern __shared__ float sdata[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y;

    float mag = 0.0f;
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int gx = -(int)gray[(y-1)*width + (x-1)] + (int)gray[(y-1)*width + (x+1)]
                 - 2*(int)gray[y*width + (x-1)]  + 2*(int)gray[y*width + (x+1)]
                 - (int)gray[(y+1)*width + (x-1)] + (int)gray[(y+1)*width + (x+1)];
        int gy = -(int)gray[(y-1)*width + (x-1)] - 2*(int)gray[(y-1)*width + x] - (int)gray[(y-1)*width + (x+1)]
                 + (int)gray[(y+1)*width + (x-1)] + 2*(int)gray[(y+1)*width + x] + (int)gray[(y+1)*width + (x+1)];
        mag = sqrtf((float)(gx*gx + gy*gy));
    }

    sdata[tid] = mag;
    __syncthreads();

    for (int s = block_threads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(&partial[blockIdx.y * gridDim.x + blockIdx.x], sdata[0]);
}

extern "C"
float launch_sobel_kernel(const uint8_t* d_gray, int width, int height, float* d_partial) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    int num_blocks = grid.x * grid.y;

    // The kernel uses atomicAdd, so the scratch buffer must start at zero.
    // The scratch is shared across kernel calls in the same image, so we
    // explicitly clear before this kernel rather than relying on the
    // (possibly stale) contents left by the brightness/contrast launchers.
    cudaMemsetAsync(d_partial, 0, num_blocks * sizeof(float));

    int shared_size = block.x * block.y * sizeof(float);
    int interior = (width > 2 && height > 2) ? (width - 2) * (height - 2) : 1;
    sobel_edge_kernel<<<grid, block, shared_size>>>(d_gray, d_partial, width, height, interior);
    cudaGetLastError();

    float* h_partial = new float[num_blocks];
    cudaMemcpy(h_partial, d_partial, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < num_blocks; i++) total += h_partial[i];
    delete[] h_partial;

    return total / (float)interior;
}
