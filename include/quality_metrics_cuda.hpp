#pragma once
#include "types.hpp"
#include "cuda_buffer.hpp"
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <string>

struct CudaStatus {
    bool ok = true;
    std::string message;
};

// Owns persistent device buffers + CUDA events for the CUDA metric pipeline.
// Every CUDA API call is checked and any failure is returned as a CudaStatus
// with a contextual message; on failure no metric fields are written to `out`.
class CudaQualityAnalyzer {
public:
    CudaQualityAnalyzer();
    ~CudaQualityAnalyzer();

    CudaQualityAnalyzer(const CudaQualityAnalyzer&)            = delete;
    CudaQualityAnalyzer& operator=(const CudaQualityAnalyzer&) = delete;
    CudaQualityAnalyzer(CudaQualityAnalyzer&&)                 = delete;
    CudaQualityAnalyzer& operator=(CudaQualityAnalyzer&&)      = delete;

    // Fills brightness, contrast, edge_score, saturated_pixel_ratio, and
    // cuda_time_ms in `out`. Returns {ok=false, message} on any CUDA failure;
    // `out` is left untouched on failure.
    CudaStatus compute(const cv::Mat& bgr, ImageMetrics& out, const Thresholds& t);

private:
    CudaBuffer  d_bgr_;
    CudaBuffer  d_gray_;
    CudaBuffer  d_float_scratch_;
    CudaBuffer  d_int_scratch_;
    cudaEvent_t ev_start_      = nullptr;
    cudaEvent_t ev_stop_       = nullptr;
    bool        ev_start_ok_   = false;
    bool        ev_stop_ok_    = false;
};
