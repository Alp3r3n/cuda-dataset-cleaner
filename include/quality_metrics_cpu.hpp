#pragma once
#include "types.hpp"
#include <opencv2/core.hpp>

// Fills brightness, contrast, edge_score, saturated_pixel_ratio, and cpu_time_ms in out.
void computeCPUMetrics(const cv::Mat& bgr, ImageMetrics& out, const Thresholds& t);
