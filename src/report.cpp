#include "report.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdio>

// Escape a string for use as a JSON string value. Handles backslash, quote,
// the common control-character escapes (\n, \r, \t, \b, \f) and emits
// \u00XX for any other ASCII control character (< 0x20).
static std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                    out += buf;
                } else {
                    out += (char)c;
                }
        }
    }
    return out;
}

// Escape a string for a CSV field per RFC 4180: if it contains a comma,
// quote, CR, or LF the field is wrapped in double quotes and any internal
// double quote is doubled.
static std::string csvEscape(const std::string& s) {
    bool needs_quote = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            needs_quote = true;
            break;
        }
    }
    if (!needs_quote) return s;
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else          out += c;
    }
    out += '"';
    return out;
}

static std::string flagsArray(const std::vector<std::string>& flags) {
    std::string out = "[";
    for (size_t i = 0; i < flags.size(); i++) {
        if (i > 0) out += ", ";
        out += "\"" + flags[i] + "\"";
    }
    out += "]";
    return out;
}

void writeJSON(const std::string& path,
               const std::vector<ImageMetrics>& results,
               const DatasetSummary& s) {
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open %s for writing\n", path.c_str());
        return;
    }

    f << std::fixed << std::setprecision(2);
    f << "{\n";
    f << "  \"summary\": {\n";
    f << "    \"total_images\": "       << s.total_images       << ",\n";
    f << "    \"clean_images\": "       << s.clean_images       << ",\n";
    f << "    \"review_images\": "      << s.review_images      << ",\n";
    f << "    \"delete_images\": "      << s.delete_images      << ",\n";
    f << "    \"blurry_count\": "       << s.blurry_count       << ",\n";
    f << "    \"too_dark_count\": "     << s.too_dark_count     << ",\n";
    f << "    \"too_bright_count\": "   << s.too_bright_count   << ",\n";
    f << "    \"low_contrast_count\": " << s.low_contrast_count << ",\n";
    f << "    \"total_cpu_time_ms\": "  << s.total_cpu_time_ms  << ",\n";
    f << "    \"total_cuda_time_ms\": " << s.total_cuda_time_ms << ",\n";
    f << "    \"speedup\": "            << s.speedup            << "\n";
    f << "  },\n";
    f << "  \"images\": [\n";

    for (size_t i = 0; i < results.size(); i++) {
        const auto& m = results[i];
        f << "    {\n";
        f << "      \"file_path\": \""    << jsonEscape(m.file_path) << "\",\n";
        if (m.load_error) {
            f << "      \"error\": \""    << jsonEscape(m.error_message) << "\"\n";
        } else {
            f << "      \"width\": "                  << m.width                 << ",\n";
            f << "      \"height\": "                 << m.height                << ",\n";
            f << "      \"brightness\": "             << m.brightness            << ",\n";
            f << "      \"contrast\": "               << m.contrast              << ",\n";
            f << "      \"edge_score\": "             << m.edge_score            << ",\n";
            f << "      \"saturated_pixel_ratio\": "  << m.saturated_pixel_ratio << ",\n";
            f << "      \"quality_score\": "          << m.quality_score         << ",\n";
            f << "      \"flags\": "                  << flagsArray(m.flags)     << ",\n";
            f << "      \"recommendation\": \""       << m.recommendation        << "\",\n";
            f << "      \"cpu_time_ms\": "            << m.cpu_time_ms           << ",\n";
            f << "      \"cuda_time_ms\": "           << m.cuda_time_ms          << "\n";
        }
        f << "    }";
        if (i + 1 < results.size()) f << ",";
        f << "\n";
    }

    f << "  ]\n";
    f << "}\n";
}

void writeCSV(const std::string& path,
              const std::vector<ImageMetrics>& results) {
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open %s for writing\n", path.c_str());
        return;
    }

    f << "file_path,width,height,brightness,contrast,edge_score,"
         "saturated_pixel_ratio,quality_score,flags,recommendation,"
         "cpu_time_ms,cuda_time_ms\n";
    f << std::fixed << std::setprecision(4);

    for (const auto& m : results) {
        std::string flags_str;
        for (size_t i = 0; i < m.flags.size(); i++) {
            if (i > 0) flags_str += ";";
            flags_str += m.flags[i];
        }
        f << csvEscape(m.file_path) << ","
          << m.width << ","
          << m.height << ","
          << m.brightness << ","
          << m.contrast << ","
          << m.edge_score << ","
          << m.saturated_pixel_ratio << ","
          << m.quality_score << ","
          << csvEscape(flags_str) << ","
          << csvEscape(m.recommendation) << ","
          << m.cpu_time_ms << ","
          << m.cuda_time_ms << "\n";
    }
}
