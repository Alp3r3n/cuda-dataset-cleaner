#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Each launcher returns the cudaError_t from the most recent failing CUDA
// API call (or cudaSuccess). Metric values are written through `out_value`.
// Scratch buffers must be sized by the caller (CudaQualityAnalyzer) — see
// `CudaQualityAnalyzer::compute()` for the contract.
extern "C" {

cudaError_t launch_grayscale_kernel(const uint8_t* d_bgr, uint8_t* d_gray,
                                    int width, int height);

cudaError_t launch_brightness_kernel(const uint8_t* d_gray, int num_pixels,
                                     float* d_partial, float* out_value);

cudaError_t launch_contrast_kernel(const uint8_t* d_gray, int num_pixels,
                                   float mean_brightness, float* d_partial,
                                   float* out_value);

cudaError_t launch_sobel_kernel(const uint8_t* d_gray, int width, int height,
                                float* d_partial, float* out_value);

cudaError_t launch_saturation_kernel(const uint8_t* d_gray, int num_pixels,
                                     float threshold, int* d_partial,
                                     float* out_value);

}
