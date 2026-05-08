#include "quality_metrics_cpu.hpp"
#include <chrono>
#include <opencv2/imgproc.hpp>

void computeCPUMetrics(const cv::Mat& bgr, ImageMetrics& out, const Thresholds& t) {
    auto t0 = std::chrono::high_resolution_clock::now();

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    cv::Scalar mean_s, stddev_s;
    cv::meanStdDev(gray, mean_s, stddev_s);
    out.brightness = static_cast<float>(mean_s[0]);
    out.contrast   = static_cast<float>(stddev_s[0]);

    cv::Mat sx, sy, mag;
    cv::Sobel(gray, sx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, sy, CV_32F, 0, 1, 3);
    cv::magnitude(sx, sy, mag);
    out.edge_score = static_cast<float>(cv::mean(mag)[0]);

    cv::Mat sat_mask = gray > t.saturation_pixel_threshold;
    int sat_count    = cv::countNonZero(sat_mask);
    int total_pixels = gray.rows * gray.cols;
    out.saturated_pixel_ratio = (total_pixels > 0)
        ? static_cast<float>(sat_count) / static_cast<float>(total_pixels)
        : 0.0f;

    auto t1 = std::chrono::high_resolution_clock::now();
    out.cpu_time_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
}
