#pragma once
#include <string>
#include <vector>

struct Thresholds {
    float too_dark_below             = 40.0f;
    float severe_dark_threshold      = 10.0f;
    float too_bright_above           = 220.0f;
    float low_contrast_below         = 25.0f;
    float blurry_below               = 5.0f;
    float saturation_pixel_threshold = 235.0f;
    float saturated_ratio_threshold  = 0.25f;
    float delete_edge_threshold      = 5.0f;
    float delete_contrast_threshold  = 15.0f;
    int   penalty_blurry             = 35;
    int   penalty_too_dark           = 25;
    int   penalty_too_bright         = 25;
    int   penalty_low_contrast       = 20;
    int   keep_score                 = 70;
    int   delete_score               = 40;
};

struct ImageMetrics {
    std::string file_path;
    int   width                  = 0;
    int   height                 = 0;
    float brightness             = 0.0f;
    float contrast               = 0.0f;
    float edge_score             = 0.0f;
    float saturated_pixel_ratio  = 0.0f;
    int   quality_score          = 0;
    std::vector<std::string> flags;
    std::string recommendation;
    float cpu_time_ms            = 0.0f;
    float cuda_time_ms           = 0.0f;
    bool  load_error             = false;
    std::string error_message;
};

struct DatasetSummary {
    int   total_images       = 0;
    int   clean_images       = 0;
    int   review_images      = 0;
    int   delete_images      = 0;
    int   blurry_count       = 0;
    int   too_dark_count     = 0;
    int   too_bright_count   = 0;
    int   low_contrast_count = 0;
    float total_cpu_time_ms  = 0.0f;
    float total_cuda_time_ms = 0.0f;
    float speedup            = 0.0f;
};
