#include "image_loader.hpp"
#include "quality_metrics_cpu.hpp"
#include "quality_metrics_cuda.hpp"
#include "benchmark.hpp"
#include "report.hpp"
#include "organizer.hpp"
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>
#include <unistd.h>

static void printUsage(const char* bin) {
    printf("Usage: %s scan <input_path> [options]\n\n"
           "Options:\n"
           "  --output <path>           JSON report path (default: report.json)\n"
           "  --csv <path>              CSV summary path (default: summary.csv)\n"
           "  --recursive               Scan subdirectories (default: on)\n"
           "  --no-recursive            Do not scan subdirectories\n"
           "  --cpu                     Run CPU metrics (default: on)\n"
           "  --no-cpu                  Skip CPU metrics\n"
           "  --cuda                    Run CUDA metrics (default: on)\n"
           "  --no-cuda                 Skip CUDA metrics\n"
           "  --config <path>           Load thresholds from JSON config\n"
           "  --preset <name>           Apply preset block from config (default|relaxed|strict)\n"
           "  --organize-output <dir>   Generate organize plan (dry-run unless --copy-files)\n"
           "  --copy-files              With --organize-output, copy files into keep/review/delete\n\n"
           "Examples:\n"
           "  %s scan ./dataset --output report.json --csv summary.csv\n"
           "  %s scan ./dataset --preset strict --organize-output ./triage\n"
           "  %s scan ./dataset --organize-output ./triage --copy-files\n",
           bin, bin, bin, bin);
}

int main(int argc, char* argv[]) {
    if (argc < 3 || std::string(argv[1]) != "scan") {
        printUsage(argv[0]);
        return 1;
    }

    std::string input_path      = argv[2];
    std::string output_path     = "report.json";
    std::string csv_path        = "summary.csv";
    std::string config_path     = "";
    std::string preset_name     = "";
    std::string organize_output = "";
    bool recursive  = true;
    bool run_cpu    = true;
    bool run_cuda   = true;
    bool copy_files = false;

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--output"           && i+1 < argc) output_path     = argv[++i];
        else if (arg == "--csv"              && i+1 < argc) csv_path        = argv[++i];
        else if (arg == "--config"           && i+1 < argc) config_path     = argv[++i];
        else if (arg == "--preset"           && i+1 < argc) preset_name     = argv[++i];
        else if (arg == "--organize-output"  && i+1 < argc) organize_output = argv[++i];
        else if (arg == "--copy-files")      copy_files = true;
        else if (arg == "--recursive")       recursive  = true;
        else if (arg == "--no-recursive")    recursive  = false;
        else if (arg == "--cpu")             run_cpu    = true;
        else if (arg == "--no-cpu")          run_cpu    = false;
        else if (arg == "--cuda")            run_cuda   = true;
        else if (arg == "--no-cuda")         run_cuda   = false;
        else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            printUsage(argv[0]);
            return 1;
        }
    }

    // Refuse destructive layouts upfront, before doing any scan work.
    if (!organize_output.empty()) {
        if (organizeOutputOverlapsInput(input_path, organize_output)) {
            fprintf(stderr,
                    "ERROR: --organize-output '%s' overlaps the input path '%s'. "
                    "Refusing to organize.\n",
                    organize_output.c_str(), input_path.c_str());
            return 2;
        }
    }

    Thresholds thresholds;
    if (!config_path.empty())
        loadThresholdsFromFile(config_path, thresholds);
    if (!preset_name.empty()) {
        // If --config wasn't supplied, fall back to the standard config.json
        // path so a bare `--preset relaxed` still works in the project dir.
        std::string preset_source = config_path.empty() ? std::string("config.json") : config_path;
        applyPreset(preset_source, preset_name, thresholds);
    }

    printf("Scanning: %s\n", input_path.c_str());
    auto image_paths = scanDataset(input_path, recursive);
    if (image_paths.empty()) {
        printf("No supported images found in: %s\n", input_path.c_str());
        return 0;
    }
    printf("Found %zu image(s). Processing...\n", image_paths.size());

    std::vector<ImageMetrics> results;
    results.reserve(image_paths.size());

    // One analyzer for the whole run — its CudaBuffers persist and grow
    // across images instead of being allocated/freed per call.
    CudaQualityAnalyzer cuda_analyzer;

    // TTY-aware progress output: in-place updates with line clear when
    // attached to a terminal, one line per image when piped/redirected.
    bool is_tty = isatty(fileno(stdout)) != 0;

    for (size_t idx = 0; idx < image_paths.size(); idx++) {
        const auto& fpath = image_paths[idx];
        if (is_tty) {
            printf("\r[%zu/%zu] %s\033[K", idx + 1, image_paths.size(), fpath.c_str());
        } else {
            printf("[%zu/%zu] %s\n", idx + 1, image_paths.size(), fpath.c_str());
        }
        fflush(stdout);

        ImageMetrics m;
        m.file_path = fpath;

        cv::Mat img = loadImage(fpath);
        if (img.empty()) {
            m.load_error = true;
            m.error_message = "Failed to load image";
            results.push_back(std::move(m));
            continue;
        }

        m.width  = img.cols;
        m.height = img.rows;

        // Track whether at least one backend produced valid metrics. If
        // neither does (CUDA fails and CPU is disabled, or both flags off),
        // we record the failure and skip flagging/scoring so default zero
        // metrics never reach applyFlagsAndScore.
        bool have_metrics = false;

        if (run_cpu) {
            computeCPUMetrics(img, m, thresholds);
            have_metrics = true;
        }

        if (run_cuda) {
            ImageMetrics cuda_m = m;
            CudaStatus status = cuda_analyzer.compute(img, cuda_m, thresholds);
            if (status.ok) {
                m.brightness            = cuda_m.brightness;
                m.contrast              = cuda_m.contrast;
                m.edge_score            = cuda_m.edge_score;
                m.saturated_pixel_ratio = cuda_m.saturated_pixel_ratio;
                m.cuda_time_ms          = cuda_m.cuda_time_ms;
                have_metrics = true;
            } else if (!have_metrics) {
                // No CPU fallback — record the failure and move on.
                m.load_error = true;
                m.error_message = "CUDA metrics failed: " + status.message;
                results.push_back(std::move(m));
                continue;
            } else {
                fprintf(stderr,
                        "WARN: CUDA metrics failed for %s: %s (keeping CPU metrics)\n",
                        fpath.c_str(), status.message.c_str());
            }
        }

        if (!have_metrics) {
            m.load_error = true;
            m.error_message = "No metric backend produced valid metrics";
            results.push_back(std::move(m));
            continue;
        }

        applyFlagsAndScore(m, thresholds);
        results.push_back(std::move(m));
    }
    if (is_tty) printf("\n");

    DatasetSummary summary = computeSummary(results);
    printSummary(summary);

    writeJSON(output_path, results, summary);
    printf("JSON report: %s\n", output_path.c_str());

    writeCSV(csv_path, results);
    printf("CSV summary: %s\n", csv_path.c_str());

    if (!organize_output.empty()) {
        OrganizeOptions opts;
        opts.output_dir  = organize_output;
        opts.copy_files  = copy_files;

        auto plan = buildOrganizePlan(input_path, results, opts);

        std::error_code ec;
        std::filesystem::create_directories(opts.output_dir, ec);
        if (ec) {
            fprintf(stderr, "ERROR: cannot create organize output dir '%s': %s\n",
                    opts.output_dir.c_str(), ec.message().c_str());
            return 3;
        }

        if (copy_files) {
            int copied = executeOrganizeCopy(plan);
            printf("Copied %d file(s) into %s/{keep,review,delete}/\n",
                   copied, opts.output_dir.c_str());
        } else {
            size_t total = 0, skipped = 0;
            for (const auto& e : plan) {
                if (e.action == "skip") skipped++;
                else total++;
            }
            printf("Organize dry-run: %zu file mapping(s) planned (%zu skipped). "
                   "Pass --copy-files to actually copy.\n", total, skipped);
        }

        std::string plan_path =
            (std::filesystem::path(opts.output_dir) / "organize_plan.csv").string();
        writeOrganizePlan(plan_path, plan);
        printf("Organize plan: %s\n", plan_path.c_str());
    }

    printf("Done.\n");

    return 0;
}
