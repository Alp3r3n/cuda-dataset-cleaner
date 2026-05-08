#pragma once
#include "types.hpp"
#include <string>
#include <vector>

void writeJSON(const std::string& path,
               const std::vector<ImageMetrics>& results,
               const DatasetSummary& summary);

void writeCSV(const std::string& path,
              const std::vector<ImageMetrics>& results);
