#pragma once
#include "types.hpp"
#include <string>
#include <vector>
#include <opencv2/core.hpp>

std::vector<std::string> scanDataset(const std::string& folder, bool recursive);
cv::Mat loadImage(const std::string& path);

void applyFlagsAndScore(ImageMetrics& m, const Thresholds& t);
void loadThresholdsFromFile(const std::string& path, Thresholds& t);
void applyPreset(const std::string& path, const std::string& preset_name, Thresholds& t);
