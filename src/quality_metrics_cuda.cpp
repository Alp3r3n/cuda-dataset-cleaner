#include "quality_metrics_cuda.hpp"
#include "kernel_launchers.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <opencv2/imgproc.hpp>

CudaQualityAnalyzer::CudaQualityAnalyzer()
    : ev_start_(nullptr), ev_stop_(nullptr) {
    cudaEventCreate(&ev_start_);
    cudaEventCreate(&ev_stop_);
}

CudaQualityAnalyzer::~CudaQualityAnalyzer() {
    if (ev_start_) cudaEventDestroy(ev_start_);
    if (ev_stop_)  cudaEventDestroy(ev_stop_);
}

static bool checkCuda(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", context, cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool CudaQualityAnalyzer::compute(const cv::Mat& bgr, ImageMetrics& out, const Thresholds& t) {
    if (bgr.empty() || bgr.type() != CV_8UC3) return false;

    int width  = bgr.cols;
    int height = bgr.rows;
    int n      = width * height;

    // ---- Untimed: ensure all device buffers are large enough ----------
    // Reduction grid sizes:
    //   1D launchers (brightness/contrast/saturation) use block_size = 256.
    //   Sobel uses 16x16 = 256 threads in a 2D grid.
    int n_blocks_1d   = (n + 255) / 256;
    int n_blocks_2d   = ((width + 15) / 16) * ((height + 15) / 16);
    int max_float_blk = std::max(n_blocks_1d, n_blocks_2d);

    d_bgr_.ensureCapacity((size_t)n * 3);
    d_gray_.ensureCapacity((size_t)n);
    d_float_scratch_.ensureCapacity((size_t)max_float_blk * sizeof(float));
    d_int_scratch_.ensureCapacity((size_t)n_blocks_1d * sizeof(int));

    if (!d_bgr_.data() || !d_gray_.data() ||
        !d_float_scratch_.data() || !d_int_scratch_.data()) {
        return false;
    }

    auto* d_bgr  = static_cast<uint8_t*>(d_bgr_.data());
    auto* d_gray = static_cast<uint8_t*>(d_gray_.data());
    auto* d_fp   = static_cast<float*>(d_float_scratch_.data());
    auto* d_ip   = static_cast<int*>(d_int_scratch_.data());

    // ---- Timed region: H2D + 5 kernel pipelines ----------------------
    cudaEventRecord(ev_start_);

    if (!checkCuda(cudaMemcpy(d_bgr, bgr.data, (size_t)n * 3, cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D")) {
        return false;
    }

    launch_grayscale_kernel(d_bgr, d_gray, width, height);
    float brightness        = launch_brightness_kernel(d_gray, n, d_fp);
    float contrast          = launch_contrast_kernel(d_gray, n, brightness, d_fp);
    float edge_score        = launch_sobel_kernel(d_gray, width, height, d_fp);
    float saturated_ratio   = launch_saturation_kernel(d_gray, n,
                                                       t.saturation_pixel_threshold, d_ip);

    cudaEventRecord(ev_stop_);
    cudaEventSynchronize(ev_stop_);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, ev_start_, ev_stop_);

    out.brightness            = brightness;
    out.contrast              = contrast;
    out.edge_score            = edge_score;
    out.saturated_pixel_ratio = saturated_ratio;
    out.cuda_time_ms          = elapsed_ms;
    return true;
}
