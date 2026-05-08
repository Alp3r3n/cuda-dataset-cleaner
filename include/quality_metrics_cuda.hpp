#pragma once
#include "types.hpp"
#include "cuda_buffer.hpp"
#include <opencv2/core.hpp>
#include <cuda_runtime.h>

// Owns persistent device buffers for the CUDA metric pipeline so memory is
// reused across images instead of allocated/freed per call. Reduction scratch
// buffers and CUDA events are also persistent — only the H2D copy and kernel
// launches happen inside the timed region.
class CudaQualityAnalyzer {
public:
    CudaQualityAnalyzer();
    ~CudaQualityAnalyzer();

    CudaQualityAnalyzer(const CudaQualityAnalyzer&)            = delete;
    CudaQualityAnalyzer& operator=(const CudaQualityAnalyzer&) = delete;
    CudaQualityAnalyzer(CudaQualityAnalyzer&&)                 = delete;
    CudaQualityAnalyzer& operator=(CudaQualityAnalyzer&&)      = delete;

    // Fills brightness, contrast, edge_score, saturated_pixel_ratio, and
    // cuda_time_ms in `out`. Returns false on CUDA error.
    bool compute(const cv::Mat& bgr, ImageMetrics& out, const Thresholds& t);

private:
    CudaBuffer  d_bgr_;
    CudaBuffer  d_gray_;
    CudaBuffer  d_float_scratch_;   // brightness / contrast / sobel partial sums
    CudaBuffer  d_int_scratch_;     // saturation counts
    cudaEvent_t ev_start_;
    cudaEvent_t ev_stop_;
};
