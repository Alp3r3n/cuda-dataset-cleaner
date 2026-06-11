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

static constexpr const char* APP_NAME    = "CudaDatasetCleaner";
static constexpr const char* APP_VERSION = "0.5";

struct ParsedArgs {
    std::string input_path;
    std::string output_path     = "report.json";
    std::string csv_path        = "summary.csv";
    std::string config_path;
    std::string preset_name;
    std::string organize_output;
    bool recursive  = true;
    bool run_cpu    = true;
    bool run_cuda   = true;
    bool copy_files = false;
};

static void printHelp() {
    printf(
        "%s v%s -- CUDA/C++ image dataset quality scanner.\n"
        "\n"
        "USAGE:\n"
        "  cuda-dataset-cleaner <command> [options]\n"
        "\n"
        "COMMANDS:\n"
        "  scan <path>                Scan an image folder and write report files.\n"
        "  help, --help, -h           Show this help message and exit.\n"
        "  version, --version, -V     Show version and exit.\n"
        "\n"
        "SCAN OPTIONS:\n"
        "  --output <path>            JSON report path (default: report.json)\n"
        "  --csv <path>               CSV summary path (default: summary.csv)\n"
        "  --recursive                Scan subdirectories (default: on)\n"
        "  --no-recursive             Do not scan subdirectories\n"
        "  --cpu                      Run CPU metrics (default: on)\n"
        "  --no-cpu                   Skip CPU metrics\n"
        "  --cuda                     Run CUDA metrics (default: on)\n"
        "  --no-cuda                  Skip CUDA metrics\n"
        "  --config <path>            Load thresholds from JSON config\n"
        "  --preset <name>            Apply preset (default | relaxed | strict) from config\n"
        "  --organize-output <dir>    Generate organize plan (dry-run unless --copy-files)\n"
        "  --copy-files               With --organize-output, copy files into keep/review/delete\n"
        "\n"
        "EXAMPLES:\n"
        "  # Basic scan, default thresholds\n"
        "  cuda-dataset-cleaner scan ./dataset --output report.json --csv summary.csv\n"
        "\n"
        "  # Apply strict preset from config.json\n"
        "  cuda-dataset-cleaner scan ./dataset --preset strict\n"
        "\n"
        "  # Dry-run organize: write organize_plan.csv only, no copies\n"
        "  cuda-dataset-cleaner scan ./dataset --organize-output ./triage\n"
        "\n"
        "  # Copy files into bucket folders (originals are never moved)\n"
        "  cuda-dataset-cleaner scan ./dataset --organize-output ./triage --copy-files\n",
        APP_NAME, APP_VERSION);
}

static void printVersion() {
    printf("%s v%s\n", APP_NAME, APP_VERSION);
    printf("CUDA/C++ image dataset quality scanner\n");
}

static void hint() {
    fprintf(stderr, "Run with --help for usage.\n");
}

static bool needValue(int i, int argc, const char* flag) {
    if (i + 1 >= argc) {
        fprintf(stderr, "ERROR: %s requires a value.\n", flag);
        hint();
        return false;
    }
    return true;
}

// argv[1] is "scan", argv[2] is input_path, argv[3..] are options.
static bool parseArgs(int argc, char* argv[], ParsedArgs& out) {
    out.input_path = argv[2];
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--output")           { if (!needValue(i, argc, "--output"))           return false; out.output_path     = argv[++i]; }
        else if (arg == "--csv")              { if (!needValue(i, argc, "--csv"))              return false; out.csv_path        = argv[++i]; }
        else if (arg == "--config")           { if (!needValue(i, argc, "--config"))           return false; out.config_path     = argv[++i]; }
        else if (arg == "--preset")           { if (!needValue(i, argc, "--preset"))           return false; out.preset_name     = argv[++i]; }
        else if (arg == "--organize-output")  { if (!needValue(i, argc, "--organize-output"))  return false; out.organize_output = argv[++i]; }
        else if (arg == "--copy-files")       out.copy_files = true;
        else if (arg == "--recursive")        out.recursive  = true;
        else if (arg == "--no-recursive")     out.recursive  = false;
        else if (arg == "--cpu")              out.run_cpu    = true;
        else if (arg == "--no-cpu")           out.run_cpu    = false;
        else if (arg == "--cuda")             out.run_cuda   = true;
        else if (arg == "--no-cuda")          out.run_cuda   = false;
        else {
            fprintf(stderr, "ERROR: unknown argument: %s\n", arg.c_str());
            hint();
            return false;
        }
    }
    return true;
}

static bool validateArgs(const ParsedArgs& a) {
    if (a.copy_files && a.organize_output.empty()) {
        fprintf(stderr, "ERROR: --copy-files requires --organize-output.\n");
        hint();
        return false;
    }
    if (!a.run_cpu && !a.run_cuda) {
        fprintf(stderr,
                "ERROR: --no-cpu and --no-cuda cannot both be set; "
                "at least one metric backend must run.\n");
        hint();
        return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "ERROR: no command given.\n");
        hint();
        return 1;
    }

    std::string a1 = argv[1];
    if (a1 == "--help"    || a1 == "-h" || a1 == "help")    { printHelp();    return 0; }
    if (a1 == "--version" || a1 == "-V" || a1 == "version") { printVersion(); return 0; }

    if (a1 != "scan") {
        fprintf(stderr, "ERROR: unknown command '%s'.\n", a1.c_str());
        hint();
        return 1;
    }

    // 'scan --help' / 'scan -h' is a help request, not an error.
    if (argc >= 3) {
        std::string a2 = argv[2];
        if (a2 == "--help" || a2 == "-h") { printHelp(); return 0; }
    }

    if (argc < 3) {
        fprintf(stderr, "ERROR: 'scan' requires an input path.\n");
        hint();
        return 1;
    }

    ParsedArgs args;
    if (!parseArgs(argc, argv, args)) return 2;
    if (!validateArgs(args))          return 2;

    // Refuse destructive layouts upfront, before any scan work.
    if (!args.organize_output.empty()) {
        if (organizeOutputOverlapsInput(args.input_path, args.organize_output)) {
            fprintf(stderr,
                    "ERROR: --organize-output '%s' overlaps the input path '%s'. "
                    "Refusing to organize.\n",
                    args.organize_output.c_str(), args.input_path.c_str());
            return 2;
        }
    }

    Thresholds thresholds;
    if (!args.config_path.empty())
        loadThresholdsFromFile(args.config_path, thresholds);
    if (!args.preset_name.empty()) {
        std::string preset_source = args.config_path.empty()
                                  ? std::string("config.json") : args.config_path;
        applyPreset(preset_source, args.preset_name, thresholds);
    }

    printf("Scanning: %s\n", args.input_path.c_str());
    auto image_paths = scanDataset(args.input_path, args.recursive);
    if (image_paths.empty()) {
        printf("No supported images found in: %s\n", args.input_path.c_str());
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

        bool have_metrics = false;

        if (args.run_cpu) {
            computeCPUMetrics(img, m, thresholds);
            have_metrics = true;
        }

        if (args.run_cuda) {
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

    writeJSON(args.output_path, results, summary);
    printf("JSON report: %s\n", args.output_path.c_str());

    writeCSV(args.csv_path, results);
    printf("CSV summary: %s\n", args.csv_path.c_str());

    if (!args.organize_output.empty()) {
        OrganizeOptions opts;
        opts.output_dir  = args.organize_output;
        opts.copy_files  = args.copy_files;

        auto plan = buildOrganizePlan(args.input_path, results, opts);

        std::error_code ec;
        std::filesystem::create_directories(opts.output_dir, ec);
        if (ec) {
            fprintf(stderr, "ERROR: cannot create organize output dir '%s': %s\n",
                    opts.output_dir.c_str(), ec.message().c_str());
            return 3;
        }

        if (args.copy_files) {
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
