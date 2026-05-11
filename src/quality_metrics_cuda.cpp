#include "quality_metrics_cuda.hpp"
#include "kernel_launchers.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <opencv2/imgproc.hpp>

CudaQualityAnalyzer::CudaQualityAnalyzer() {
    ev_start_ok_ = (cudaEventCreate(&ev_start_) == cudaSuccess);
    ev_stop_ok_  = (cudaEventCreate(&ev_stop_)  == cudaSuccess);
    if (!ev_start_ok_) ev_start_ = nullptr;
    if (!ev_stop_ok_)  ev_stop_  = nullptr;
}

CudaQualityAnalyzer::~CudaQualityAnalyzer() {
    if (ev_start_ok_) cudaEventDestroy(ev_start_);
    if (ev_stop_ok_)  cudaEventDestroy(ev_stop_);
}

static CudaStatus statusFromErr(cudaError_t err, const char* what) {
    return {false, std::string(what) + ": " + cudaGetErrorString(err)};
}

#define CUDA_CHECK(call, what) \
    do { cudaError_t _err = (call); \
         if (_err != cudaSuccess) return statusFromErr(_err, what); } while (0)

CudaStatus CudaQualityAnalyzer::compute(const cv::Mat& bgr, ImageMetrics& out, const Thresholds& t) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        return {false, "invalid input image (empty or not CV_8UC3)"};
    }
    if (!ev_start_ok_ || !ev_stop_ok_) {
        return {false, "CUDA events were not created successfully"};
    }

    int width  = bgr.cols;
    int height = bgr.rows;
    int n      = width * height;

    // Untimed: ensure all device buffers are large enough.
    int n_blocks_1d   = (n + 255) / 256;
    int n_blocks_2d   = ((width + 15) / 16) * ((height + 15) / 16);
    int max_float_blk = std::max(n_blocks_1d, n_blocks_2d);

    if (!d_bgr_.ensureCapacity((size_t)n * 3))
        return {false, "d_bgr ensureCapacity failed"};
    if (!d_gray_.ensureCapacity((size_t)n))
        return {false, "d_gray ensureCapacity failed"};
    if (!d_float_scratch_.ensureCapacity((size_t)max_float_blk * sizeof(float)))
        return {false, "d_float_scratch ensureCapacity failed"};
    if (!d_int_scratch_.ensureCapacity((size_t)n_blocks_1d * sizeof(int)))
        return {false, "d_int_scratch ensureCapacity failed"};

    auto* d_bgr  = static_cast<uint8_t*>(d_bgr_.data());
    auto* d_gray = static_cast<uint8_t*>(d_gray_.data());
    auto* d_fp   = static_cast<float*>(d_float_scratch_.data());
    auto* d_ip   = static_cast<int*>(d_int_scratch_.data());

    // Timed region: H2D + 5 kernel pipelines.
    CUDA_CHECK(cudaEventRecord(ev_start_), "cudaEventRecord(start)");

    CUDA_CHECK(cudaMemcpy(d_bgr, bgr.data, (size_t)n * 3, cudaMemcpyHostToDevice),
               "cudaMemcpy H2D");

    CUDA_CHECK(launch_grayscale_kernel(d_bgr, d_gray, width, height),
               "grayscale kernel");

    float brightness = 0.0f;
    CUDA_CHECK(launch_brightness_kernel(d_gray, n, d_fp, &brightness),
               "brightness kernel");

    float contrast = 0.0f;
    CUDA_CHECK(launch_contrast_kernel(d_gray, n, brightness, d_fp, &contrast),
               "contrast kernel");

    float edge_score = 0.0f;
    CUDA_CHECK(launch_sobel_kernel(d_gray, width, height, d_fp, &edge_score),
               "sobel kernel");

    float saturated_ratio = 0.0f;
    CUDA_CHECK(launch_saturation_kernel(d_gray, n, t.saturation_pixel_threshold,
                                        d_ip, &saturated_ratio),
               "saturation kernel");

    CUDA_CHECK(cudaEventRecord(ev_stop_), "cudaEventRecord(stop)");
    CUDA_CHECK(cudaEventSynchronize(ev_stop_), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start_, ev_stop_),
               "cudaEventElapsedTime");

    out.brightness            = brightness;
    out.contrast              = contrast;
    out.edge_score            = edge_score;
    out.saturated_pixel_ratio = saturated_ratio;
    out.cuda_time_ms          = elapsed_ms;
    return {true, ""};
}
