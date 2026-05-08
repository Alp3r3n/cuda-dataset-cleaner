#pragma once
#include "types.hpp"
#include <vector>

DatasetSummary computeSummary(const std::vector<ImageMetrics>& results);
void printSummary(const DatasetSummary& summary);
