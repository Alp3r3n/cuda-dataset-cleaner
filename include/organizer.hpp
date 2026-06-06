#pragma once
#include "types.hpp"
#include <string>
#include <vector>

struct OrganizeOptions {
    std::string output_dir;
    bool        copy_files = false;   // false = dry-run plan only
};

struct OrganizePlanEntry {
    std::string source_path;
    std::string destination_path;
    std::string recommendation;       // "keep" / "review" / "delete"
    std::string action;               // "copy" | "dry-run-copy" | "skip"
};

// True if `output_dir` is the same as `input_dir`, lives inside `input_dir`,
// or contains `input_dir`. Used as the v0.4 safety refusal check.
bool organizeOutputOverlapsInput(const std::string& input_dir, const std::string& output_dir);

// Map each successfully scanned image to its bucketed destination.
// load_error images are emitted with action="skip" and no destination.
std::vector<OrganizePlanEntry> buildOrganizePlan(
    const std::string& input_dir,
    const std::vector<ImageMetrics>& results,
    const OrganizeOptions& opts);

// Copy files per plan. Creates parent directories as needed. Returns the
// number of successful copies. Originals are never modified or moved.
int executeOrganizeCopy(const std::vector<OrganizePlanEntry>& plan);

// Write <plan_path>/organize_plan.csv with columns:
// source_path,destination_path,recommendation,action
void writeOrganizePlan(const std::string& plan_path,
                       const std::vector<OrganizePlanEntry>& plan);
