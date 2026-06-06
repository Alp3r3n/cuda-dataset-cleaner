#include "organizer.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace fs = std::filesystem;

static fs::path safeCanonical(const std::string& p) {
    std::error_code ec;
    fs::path canon = fs::weakly_canonical(fs::path(p), ec);
    if (ec || canon.empty()) {
        canon = fs::absolute(fs::path(p)).lexically_normal();
    }
    return canon;
}

bool organizeOutputOverlapsInput(const std::string& input_dir, const std::string& output_dir) {
    fs::path in_can  = safeCanonical(input_dir);
    fs::path out_can = safeCanonical(output_dir);

    std::string in_s  = in_can.string();
    std::string out_s = out_can.string();
    if (in_s.empty() || out_s.empty()) return false;

    if (in_s == out_s) return true;

    auto starts_with_path = [](const std::string& haystack, const std::string& prefix) {
        if (haystack.size() <= prefix.size()) return false;
        if (haystack.compare(0, prefix.size(), prefix) != 0) return false;
        char sep = haystack[prefix.size()];
        return sep == '/' || sep == '\\';
    };

    if (starts_with_path(out_s, in_s)) return true;   // output inside input
    if (starts_with_path(in_s, out_s)) return true;   // input inside output
    return false;
}

std::vector<OrganizePlanEntry> buildOrganizePlan(
    const std::string& input_dir,
    const std::vector<ImageMetrics>& results,
    const OrganizeOptions& opts) {

    std::vector<OrganizePlanEntry> plan;
    plan.reserve(results.size());

    fs::path out_root(opts.output_dir);
    fs::path in_root = fs::path(input_dir).lexically_normal();
    const char* action_str = opts.copy_files ? "copy" : "dry-run-copy";

    for (const auto& m : results) {
        OrganizePlanEntry e;
        e.source_path    = m.file_path;
        e.recommendation = m.recommendation;

        if (m.load_error || m.recommendation.empty()) {
            e.destination_path = "";
            e.action           = "skip";
            plan.push_back(std::move(e));
            continue;
        }

        std::error_code ec;
        fs::path rel = fs::relative(fs::path(m.file_path), in_root, ec);
        if (ec || rel.empty() || rel.string().rfind("..", 0) == 0) {
            // Fallback to leaf filename if the source is outside the input root
            rel = fs::path(m.file_path).filename();
        }
        fs::path dest = out_root / m.recommendation / rel;
        e.destination_path = dest.lexically_normal().string();
        e.action           = action_str;
        plan.push_back(std::move(e));
    }
    return plan;
}

int executeOrganizeCopy(const std::vector<OrganizePlanEntry>& plan) {
    int copied = 0;
    for (const auto& e : plan) {
        if (e.action != "copy") continue;
        std::error_code ec;
        fs::path dest(e.destination_path);
        fs::create_directories(dest.parent_path(), ec);
        if (ec) {
            fprintf(stderr, "WARN: create_directories failed for %s: %s\n",
                    dest.parent_path().string().c_str(), ec.message().c_str());
            continue;
        }
        fs::copy_file(fs::path(e.source_path), dest,
                      fs::copy_options::overwrite_existing, ec);
        if (ec) {
            fprintf(stderr, "WARN: copy failed %s -> %s: %s\n",
                    e.source_path.c_str(), e.destination_path.c_str(), ec.message().c_str());
            continue;
        }
        copied++;
    }
    return copied;
}

static std::string csvEscape(const std::string& s) {
    bool needs_quote = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') { needs_quote = true; break; }
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

void writeOrganizePlan(const std::string& plan_path,
                       const std::vector<OrganizePlanEntry>& plan) {
    std::ofstream f(plan_path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open %s for writing\n", plan_path.c_str());
        return;
    }
    f << "source_path,destination_path,recommendation,action\n";
    for (const auto& e : plan) {
        f << csvEscape(e.source_path) << ","
          << csvEscape(e.destination_path) << ","
          << csvEscape(e.recommendation) << ","
          << csvEscape(e.action) << "\n";
    }
}
