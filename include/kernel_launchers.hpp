#pragma once
#include <cstdint>

// Launchers no longer allocate or free their own scratch buffers.
// The caller (CudaQualityAnalyzer) owns persistent device-side scratch
// memory and passes a pointer in. Sizing requirements:
//   d_partial (float) must hold at least max(grid_1d, grid_2d) floats, where
//     grid_1d = ceil(num_pixels / 256)
//     grid_2d = ceil(width / 16) * ceil(height / 16)
//   d_partial (int)   must hold at least grid_1d ints.
extern "C" {

void  launch_grayscale_kernel(const uint8_t* d_bgr, uint8_t* d_gray,
                              int width, int height);

float launch_brightness_kernel(const uint8_t* d_gray, int num_pixels,
                               float* d_partial);

float launch_contrast_kernel(const uint8_t* d_gray, int num_pixels,
                             float mean_brightness, float* d_partial);

float launch_sobel_kernel(const uint8_t* d_gray, int width, int height,
                          float* d_partial);

float launch_saturation_kernel(const uint8_t* d_gray, int num_pixels,
                               float threshold, int* d_partial);

}
