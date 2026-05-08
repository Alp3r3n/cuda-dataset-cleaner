#include "image_loader.hpp"
#include "quality_metrics_cpu.hpp"
#include "quality_metrics_cuda.hpp"
#include "benchmark.hpp"
#include "report.hpp"
#include <cstdio>
#include <string>
#include <vector>
#include <unistd.h>

static void printUsage(const char* bin) {
    printf("Usage: %s scan <input_path> [options]\n\n"
           "Options:\n"
           "  --output <path>   JSON report path (default: report.json)\n"
           "  --csv <path>      CSV summary path (default: summary.csv)\n"
           "  --recursive       Scan subdirectories (default: on)\n"
           "  --no-recursive    Do not scan subdirectories\n"
           "  --cpu             Run CPU metrics (default: on)\n"
           "  --no-cpu          Skip CPU metrics\n"
           "  --cuda            Run CUDA metrics (default: on)\n"
           "  --no-cuda         Skip CUDA metrics\n"
           "  --config <path>   Load thresholds from JSON config\n\n"
           "Examples:\n"
           "  %s scan ./dataset --output report.json --csv summary.csv\n"
           "  %s scan ./dataset --no-recursive --no-cuda\n",
           bin, bin, bin);
}

int main(int argc, char* argv[]) {
    if (argc < 3 || std::string(argv[1]) != "scan") {
        printUsage(argv[0]);
        return 1;
    }

    std::string input_path   = argv[2];
    std::string output_path  = "report.json";
    std::string csv_path     = "summary.csv";
    std::string config_path  = "";
    bool recursive  = true;
    bool run_cpu    = true;
    bool run_cuda   = true;

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--output"       && i+1 < argc) output_path = argv[++i];
        else if (arg == "--csv"          && i+1 < argc) csv_path    = argv[++i];
        else if (arg == "--config"       && i+1 < argc) config_path = argv[++i];
        else if (arg == "--recursive")    recursive  = true;
        else if (arg == "--no-recursive") recursive  = false;
        else if (arg == "--cpu")          run_cpu    = true;
        else if (arg == "--no-cpu")       run_cpu    = false;
        else if (arg == "--cuda")         run_cuda   = true;
        else if (arg == "--no-cuda")      run_cuda   = false;
        else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            printUsage(argv[0]);
            return 1;
        }
    }

    Thresholds thresholds;
    if (!config_path.empty())
        loadThresholdsFromFile(config_path, thresholds);

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

        if (run_cpu)  computeCPUMetrics(img, m, thresholds);
        if (run_cuda) {
            ImageMetrics cuda_m = m;
            if (cuda_analyzer.compute(img, cuda_m, thresholds)) {
                // Use CUDA metrics as primary; keep CPU time from run_cpu pass
                m.brightness            = cuda_m.brightness;
                m.contrast              = cuda_m.contrast;
                m.edge_score            = cuda_m.edge_score;
                m.saturated_pixel_ratio = cuda_m.saturated_pixel_ratio;
                m.cuda_time_ms          = cuda_m.cuda_time_ms;
            }
        }
        if (!run_cuda && !run_cpu) {
            m.load_error = true;
            m.error_message = "Both --no-cpu and --no-cuda specified";
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

    printf("Done.\n");

    return 0;
}
