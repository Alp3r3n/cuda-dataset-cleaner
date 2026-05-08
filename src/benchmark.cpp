#include "benchmark.hpp"
#include <cstdio>
#include <algorithm>

DatasetSummary computeSummary(const std::vector<ImageMetrics>& results) {
    DatasetSummary s{};
    for (const auto& m : results) {
        if (m.load_error) continue;
        s.total_images++;
        s.total_cpu_time_ms  += m.cpu_time_ms;
        s.total_cuda_time_ms += m.cuda_time_ms;
        for (const auto& f : m.flags) {
            if      (f == "blurry")        s.blurry_count++;
            else if (f == "too_dark")      s.too_dark_count++;
            else if (f == "too_bright")    s.too_bright_count++;
            else if (f == "low_contrast")  s.low_contrast_count++;
        }
        if      (m.recommendation == "keep")   s.clean_images++;
        else if (m.recommendation == "review") s.review_images++;
        else if (m.recommendation == "delete") s.delete_images++;
    }
    if (s.total_cuda_time_ms > 0.001f)
        s.speedup = s.total_cpu_time_ms / s.total_cuda_time_ms;
    return s;
}

void printSummary(const DatasetSummary& s) {
    printf("\n=== Dataset Scan Summary ===\n");
    printf("Total images  : %d\n", s.total_images);
    printf("Keep (clean)  : %d\n", s.clean_images);
    printf("Review        : %d\n", s.review_images);
    printf("Delete        : %d\n", s.delete_images);
    printf("Blurry        : %d\n", s.blurry_count);
    printf("Too dark      : %d\n", s.too_dark_count);
    printf("Too bright    : %d\n", s.too_bright_count);
    printf("Low contrast  : %d\n", s.low_contrast_count);
    printf("---\n");
    printf("CPU  total    : %.2f ms\n", s.total_cpu_time_ms);
    printf("CUDA total    : %.2f ms\n", s.total_cuda_time_ms);
    if (s.speedup > 0.0f)
        printf("Speedup       : %.2fx\n", s.speedup);
    printf("============================\n\n");
}
