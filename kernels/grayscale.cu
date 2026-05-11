#include "kernel_launchers.hpp"
#include <cuda_runtime.h>
#include <cstdint>

__global__ void bgr_to_gray_kernel(const uint8_t* bgr, uint8_t* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int px   = y * width + x;
    int bpx  = px * 3;
    float b  = (float)bgr[bpx];
    float g  = (float)bgr[bpx + 1];
    float r  = (float)bgr[bpx + 2];
    // ITU-R BT.601 coefficients
    gray[px] = (uint8_t)(0.114f * b + 0.587f * g + 0.299f * r);
}

extern "C"
cudaError_t launch_grayscale_kernel(const uint8_t* d_bgr, uint8_t* d_gray,
                                    int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    bgr_to_gray_kernel<<<grid, block>>>(d_bgr, d_gray, width, height);
    return cudaGetLastError();
}
