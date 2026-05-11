#include "image_loader.hpp"
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstring>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

static const std::vector<std::string> SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"};

static bool isSupportedImage(const fs::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    for (const auto& e : SUPPORTED_EXTS)
        if (ext == e) return true;
    return false;
}

std::vector<std::string> scanDataset(const std::string& folder, bool recursive) {
    std::vector<std::string> paths;
    if (!fs::exists(folder) || !fs::is_directory(folder))
        return paths;

    if (recursive) {
        for (const auto& entry : fs::recursive_directory_iterator(folder)) {
            if (entry.is_regular_file() && isSupportedImage(entry.path()))
                paths.push_back(entry.path().string());
        }
    } else {
        for (const auto& entry : fs::directory_iterator(folder)) {
            if (entry.is_regular_file() && isSupportedImage(entry.path()))
                paths.push_back(entry.path().string());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

cv::Mat loadImage(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    return img;
}

void applyFlagsAndScore(ImageMetrics& m, const Thresholds& t) {
    int score = 100;
    bool has_blurry       = false;
    bool has_too_dark     = false;
    bool has_too_bright   = false;
    bool has_low_contrast = false;

    if (m.edge_score < t.blurry_below) {
        m.flags.push_back("blurry");
        score -= t.penalty_blurry;
        has_blurry = true;
    }
    if (m.brightness < t.too_dark_below) {
        m.flags.push_back("too_dark");
        score -= t.penalty_too_dark;
        has_too_dark = true;
    }
    if (m.contrast < t.low_contrast_below) {
        m.flags.push_back("low_contrast");
        score -= t.penalty_low_contrast;
        has_low_contrast = true;
    }
    // too_bright now requires high mean brightness AND high saturated pixel ratio
    // AND either low contrast or low edge detail. This avoids penalizing bright but
    // detailed images (e.g. sky photos with sharp foreground objects).
    bool too_bright_brightness = m.brightness > t.too_bright_above;
    bool too_bright_saturation = m.saturated_pixel_ratio > t.saturated_ratio_threshold;
    bool too_bright_low_detail = (m.contrast < t.low_contrast_below) || (m.edge_score < t.blurry_below);
    if (too_bright_brightness && too_bright_saturation && too_bright_low_detail) {
        m.flags.push_back("too_bright");
        score -= t.penalty_too_bright;
        has_too_bright = true;
    }

    m.quality_score = std::max(0, score);

    // Delete is conservative and never triggered by brightness alone. A dark
    // image only counts as unrecoverable when it is severely dark AND very
    // low-contrast AND has almost no edge detail simultaneously. Bright
    // images need all three of {too_bright, low_contrast, blurry} to delete.
    // Anything else with at least one warning flag drops to "review".
    bool delete_dark = (m.brightness < t.severe_dark_threshold) &&
                       (m.contrast   < t.delete_contrast_threshold) &&
                       (m.edge_score < t.delete_edge_threshold);
    bool delete_bright = has_too_bright && has_low_contrast && has_blurry;

    if      (delete_dark)        m.recommendation = "delete";
    else if (delete_bright)      m.recommendation = "delete";
    else if (!m.flags.empty())   m.recommendation = "review";
    else                          m.recommendation = "keep";
}

static float extractFloat(const std::string& content, const std::string& key, float default_val) {
    std::string search = "\"" + key + "\"";
    size_t pos = content.find(search);
    if (pos == std::string::npos) return default_val;
    pos = content.find(':', pos + search.size());
    if (pos == std::string::npos) return default_val;
    pos++;
    while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
    try { return std::stof(content.substr(pos)); }
    catch (...) { return default_val; }
}

void loadThresholdsFromFile(const std::string& path, Thresholds& t) {
    std::ifstream f(path);
    if (!f.is_open()) return;
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    t.too_dark_below             = extractFloat(content, "too_dark_if_less_than",         t.too_dark_below);
    t.severe_dark_threshold      = extractFloat(content, "severe_dark_threshold",          t.severe_dark_threshold);
    t.too_bright_above           = extractFloat(content, "too_bright_if_greater_than",     t.too_bright_above);
    t.low_contrast_below         = extractFloat(content, "low_contrast_if_less_than",      t.low_contrast_below);
    t.blurry_below               = extractFloat(content, "blurry_if_edge_score_less_than", t.blurry_below);
    t.saturation_pixel_threshold = extractFloat(content, "saturation_pixel_threshold",     t.saturation_pixel_threshold);
    t.saturated_ratio_threshold  = extractFloat(content, "saturated_ratio_threshold",      t.saturated_ratio_threshold);
    t.delete_edge_threshold      = extractFloat(content, "delete_edge_threshold",          t.delete_edge_threshold);
    t.delete_contrast_threshold  = extractFloat(content, "delete_contrast_threshold",      t.delete_contrast_threshold);
    t.penalty_blurry             = (int)extractFloat(content, "penalty_blurry",            (float)t.penalty_blurry);
    t.penalty_too_dark           = (int)extractFloat(content, "penalty_too_dark",          (float)t.penalty_too_dark);
    t.penalty_too_bright         = (int)extractFloat(content, "penalty_too_bright",        (float)t.penalty_too_bright);
    t.penalty_low_contrast       = (int)extractFloat(content, "penalty_low_contrast",      (float)t.penalty_low_contrast);
}
