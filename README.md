# CudaDatasetCleaner

CudaDatasetCleaner is a CUDA/C++ command-line tool for fast, first-pass quality scanning of image datasets. It reads a folder of images, computes per-image quality metrics on both CPU and GPU, and produces JSON and CSV reports with `keep` / `review` / `delete` recommendations.

It is a **dataset triage** tool, not an automatic deletion tool. Potentially problematic images are routed to `review` so a human can take the final call. `delete` is reserved for clearly unusable extreme cases — near-black, near-flat, detail-free images, or fully saturated flat images — and the recommendation logic is deliberately conservative so that semantically usable low-light or low-contrast images are not thrown away.

## Why this project exists

Training ML/CV models on noisy datasets degrades model quality, but reviewing tens of thousands of images by hand is impractical. CudaDatasetCleaner gives a fast first pass over a dataset, surfacing only the small subset that needs human attention. The CUDA implementation also serves as a clean, end-to-end CPU-vs-GPU benchmarking case study for image-quality metrics.

## Features

- Recursive image folder scanning (JPG, JPEG, PNG, BMP)
- CPU baseline metrics computed on the host using OpenCV helpers
- Custom CUDA kernels for GPU-side metrics:
  - Grayscale conversion
  - Brightness reduction
  - Contrast / standard deviation reduction
  - Sobel edge score
  - Saturated pixel ratio
- Per-image flags: `blurry`, `too_dark`, `too_bright`, `low_contrast`
- Per-image quality score (0–100) for diagnostic reporting
- Conservative `keep` / `review` / `delete` recommendations
- JSON report and CSV summary
- CPU vs CUDA timing per image and aggregate speedup
- Configurable thresholds via `config.json`

## How It Works

Each image flows through a fixed pipeline:

1. **Load** — `cv::imread` decodes the file into a host BGR `uint8_t` buffer.
2. **CPU pass** (optional, `--cpu`) — OpenCV helpers compute reference metrics on the host.
3. **CUDA pass** (optional, `--cuda`) — the BGR buffer is copied to device memory once, then five custom kernels run in sequence: `grayscale → brightness → contrast → sobel → saturation`. CUDA event timing brackets the entire device pipeline including the H→D copy.
4. **Flag & score** — thresholds in `config.json` translate raw metrics into flags, a 0–100 quality score, and a final recommendation.
5. **Report** — per-image rows are written to JSON and CSV; an aggregate summary is printed to the terminal.

### OpenCV vs CUDA

OpenCV is used on the host side for image loading and CPU baseline helpers. The CUDA path does not use OpenCV CUDA modules; all GPU-side metric computations are implemented with custom CUDA kernels.

Concretely:

- **OpenCV** decodes image files (JPG, PNG, BMP) and provides the CPU-side image buffer.
- The **CPU baseline** computes reference metrics on the host using OpenCV's vectorized helpers (`cv::meanStdDev`, `cv::Sobel`, `cv::countNonZero`).
- The **CUDA path** copies the same image buffer to device memory and computes the same metrics using custom kernels written from scratch.

The project's focus is CUDA kernel implementation and CPU-vs-GPU benchmarking on a real workload; image codec development is explicitly out of scope.

## Build Instructions

### Prerequisites

```bash
sudo apt install build-essential cmake libopencv-dev
# CUDA Toolkit must also be installed: https://developer.nvidia.com/cuda-downloads
```

The build requires CMake 3.24 or newer and a CUDA-capable GPU.

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The binary will be at `build/cuda-dataset-cleaner`.

### GPU Architecture

By default the build uses `CMAKE_CUDA_ARCHITECTURES native` to auto-detect the installed GPU. To target a specific architecture, edit `CMakeLists.txt`:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 86)   # RTX 3000 series / Ampere desktop GPUs
# set(CMAKE_CUDA_ARCHITECTURES 89) # RTX 4000 series / Ada desktop GPUs
# set(CMAKE_CUDA_ARCHITECTURES 87) # Jetson Orin
# set(CMAKE_CUDA_ARCHITECTURES 72) # Jetson Xavier
```

## Usage

```bash
# Basic scan
./build/cuda-dataset-cleaner scan ./test_dataset --output report.json --csv summary.csv

# Scan a real-world dataset
./build/cuda-dataset-cleaner scan ~/datasets/coco/val2017 \
    --output reports/coco_val2017_v02_report.json \
    --csv    reports/coco_val2017_v02_summary.csv

# CPU only, no GPU
./build/cuda-dataset-cleaner scan ./dataset --no-cuda

# Custom thresholds
./build/cuda-dataset-cleaner scan ./dataset --config config.json

# Non-recursive scan
./build/cuda-dataset-cleaner scan ./dataset --no-recursive
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `scan <path>` | required | Path to image folder |
| `--output <path>` | `report.json` | JSON report output path |
| `--csv <path>` | `summary.csv` | CSV summary output path |
| `--recursive` | on | Scan subdirectories recursively |
| `--no-recursive` | — | Disable recursive scan |
| `--cpu` | on | Run CPU metrics |
| `--no-cpu` | — | Skip CPU metrics |
| `--cuda` | on | Run CUDA metrics |
| `--no-cuda` | — | Skip CUDA metrics |
| `--config <path>` | — | Load thresholds from JSON config |

## Example Report Output

### Terminal

```
Scanning: ./test_dataset
Found 6 image(s). Processing...

=== Dataset Scan Summary ===
Total images  : 6
Keep (clean)  : 1
Review        : 4
Delete        : 1
Blurry        : 4
Too dark      : 1
Too bright    : 1
Low contrast  : 4
---
CPU  total    : 23.42 ms
CUDA total    : 2.74 ms
Speedup       : 8.55x
============================

JSON report: report.json
CSV summary: summary.csv
```

### `report.json`

```json
{
  "summary": {
    "total_images": 5000,
    "clean_images": 4909,
    "review_images": 91,
    "delete_images": 0,
    "blurry_count": 0,
    "too_dark_count": 42,
    "too_bright_count": 0,
    "low_contrast_count": 54,
    "total_cpu_time_ms": 8804.11,
    "total_cuda_time_ms": 1469.41,
    "speedup": 5.99
  },
  "images": [
    {
      "file_path": "val2017/000000397133.jpg",
      "width": 640,
      "height": 480,
      "brightness": 134.7,
      "contrast": 51.8,
      "edge_score": 17.2,
      "saturated_pixel_ratio": 0.018,
      "quality_score": 100,
      "flags": [],
      "recommendation": "keep",
      "cpu_time_ms": 1.42,
      "cuda_time_ms": 0.51
    },
    {
      "file_path": "val2017/000000079837.jpg",
      "width": 640,
      "height": 425,
      "brightness": 27.8,
      "contrast": 24.1,
      "edge_score": 11.3,
      "saturated_pixel_ratio": 0.0,
      "quality_score": 75,
      "flags": ["too_dark"],
      "recommendation": "review",
      "cpu_time_ms": 1.31,
      "cuda_time_ms": 0.49
    }
  ]
}
```

The CSV columns are:

```
file_path,width,height,brightness,contrast,edge_score,saturated_pixel_ratio,quality_score,flags,recommendation,cpu_time_ms,cuda_time_ms
```

The `flags` column uses `;` as an inner separator so the CSV remains valid; most spreadsheet tools open it correctly.

## How Metrics Are Calculated

### Brightness
Average grayscale pixel intensity (0–255).
- **CPU**: `cv::meanStdDev` on the grayscale image
- **CUDA**: parallel block reduction of pixel sums → divide by pixel count

### Contrast
Standard deviation of grayscale pixel intensities.
- **CPU**: `cv::meanStdDev`
- **CUDA**: two-pass block reduction — first compute mean, then variance, then `sqrt(variance)`

### Edge Score (Sharpness)
Average Sobel edge magnitude. Higher = sharper.
- **CPU**: `cv::Sobel` + `cv::magnitude` + `cv::mean`
- **CUDA**: per-pixel Sobel Gx/Gy in a 2D grid, block reduction of magnitudes, divide by interior pixel count

### Saturated Pixel Ratio
Fraction of grayscale pixels above a saturation threshold (default 235/255). Used to disambiguate "uniformly blown-out" images from "bright but detailed" images.
- **CPU**: `cv::countNonZero(gray > threshold) / total_pixels`
- **CUDA**: per-pixel comparison + parallel block reduction of 0/1 counts

### Quality Score (0–100)
Starts at 100, penalized when warning flags fire:

| Flag | Detection rule | Penalty |
|---|---|---|
| `blurry` | `edge_score < 5.0` | −35 |
| `too_dark` | `brightness < 40.0` | −25 |
| `too_bright` | `brightness > 220.0` AND `saturated_pixel_ratio > 0.25` AND (`contrast < 25.0` OR `edge_score < 5.0`) | −25 |
| `low_contrast` | `contrast < 25.0` | −20 |

`too_bright` deliberately requires three conditions so that bright but detailed images (e.g. sunny outdoor scenes with sharp foreground objects) are not penalized.

The `quality_score` is still computed and reported, but the **recommendation does not rely only on `quality_score`** — it is driven by flag combinations (see below).

### Recommendation Logic

Evaluated in order:

1. `brightness < severe_dark_threshold` (default 10) AND `contrast < delete_contrast_threshold` (default 15) AND `edge_score < delete_edge_threshold` (default 5) → `delete`
2. `too_bright` AND `low_contrast` AND `blurry` → `delete`
3. Any warning flag present → `review`
4. Otherwise → `keep`

Behavioral guarantees:

- **Brightness alone never triggers delete.** A dark image only counts as unrecoverable when it is simultaneously near-black, near-flat, *and* effectively detail-free.
- **Low-light images with usable contrast or detail go to `review`, not `delete`.** A STOP sign at night, a silhouette in front of a dim laptop screen, or any other dark-but-recognizable subject stays in the review pile.
- **Any warning flag prevents `keep`.** An image with even one flag is routed to `review` so a human can confirm.
- **`delete` is reserved for clearly unusable extreme cases**: near-black flat images with no edge content, or fully blown-out flat images.

All thresholds are configurable via `config.json`.

## CPU vs CUDA Benchmark

### COCO val2017 (5,000 images)

| Metric | Value |
|---|---:|
| Total images | 5,000 |
| Keep | 4,909 |
| Review | 91 |
| Delete | 0 |
| Blurry | 0 |
| Too dark | 42 |
| Too bright | 0 |
| Low contrast | 54 |
| CPU total time | 8804.11 ms |
| CUDA total time | 1469.41 ms |
| Speedup | 5.99x |

COCO val2017 is a curated dataset, so most images are expected to be classified as `keep`. CudaDatasetCleaner uses conservative recommendations: potentially problematic images are routed to `review`, while `delete` is reserved for clearly unusable extreme cases. On this run, none of COCO val2017 met the strict delete criteria.

> Benchmark numbers may vary depending on GPU, image sizes, driver version, system load, and CUDA memory allocation overhead. The current implementation reuses device buffers and scratch buffers, but still processes images one by one. Future batching and pinned/asynchronous transfers may further improve throughput.

## Extreme Sanity Test

A small synthetic set of degenerate images is used to confirm that the conservative recommendation logic still identifies clearly unusable extreme cases.

| Image | Recommendation |
|---|---|
| `pure_black.bmp` | `delete` |
| `near_black.bmp` | `delete` |
| `pure_white.bmp` | `delete` |

| Metric | Value |
|---:|---:|
| CPU total time | 7.86 ms |
| CUDA total time | 2.55 ms |
| Speedup | 3.08× |

This verifies that conservative thresholds have not over-corrected away from the original goal: extreme garbage is still caught.

## Validation

`compute-sanitizer` (CUDA's runtime memory & race-condition checker) was run against the smoke-test dataset and reported:

```
ERROR SUMMARY: 0 errors
```

This is a smoke-level check, not a full formal verification. It helps catch common CUDA runtime memory issues on the tested inputs.

## Recent Changes

### v0.4 — Safe organize mode + presets

Three new CLI flags, all opt-in. Default behavior (`scan` with no organize flags) is unchanged.

- `--preset <name>` — applies a preset block (`default` / `relaxed` / `strict`) from `config.json`. Presets are nested under a top-level `"presets"` object and override individual thresholds. If `--config` is omitted, `./config.json` is used as the preset source.
- `--organize-output <dir>` — writes an `organize_plan.csv` mapping each scanned image to `<dir>/{keep,review,delete}/<relative-path>`. **Without** `--copy-files` this is dry-run only — no image files are copied.
- `--copy-files` — combined with `--organize-output`, actually copies files into the bucket subdirectories, preserving the source's relative folder structure. Original files are never moved or deleted.

If the output directory overlaps the input directory (same path, output inside input, or input inside output), the tool refuses with exit code `2` before any scan or filesystem work. Plan CSV columns: `source_path,destination_path,recommendation,action`.

### v0.3.2 — Reliability & validation cleanup

- Every CUDA API call inside `CudaQualityAnalyzer::compute()` is now checked and surfaced through a `CudaStatus { bool ok; std::string message; }` return type. Kernel launchers return `cudaError_t` and write their result through an out-parameter.
- A failed CUDA run no longer produces zero-valued metrics that get scored: `main.cpp` tracks `have_metrics` and routes the failure into the existing `load_error` / `error_message` fields instead of running `applyFlagsAndScore` on defaults.
- `CudaBuffer::allocate` / `ensureCapacity` now return `bool` so allocation failures propagate.
- CPU `edge_score` now averages over the same interior region as the CUDA kernel (skipping the 1-pixel Sobel border) — small parity discrepancy removed.
- JSON string escaping handles `\n`, `\r`, `\t`, `\b`, `\f`, and `\u00XX` for other control characters. CSV writing now uses RFC 4180 quoting (only quote when needed; double internal quotes).
- Unused config keys `keep_if_score_gte` and `delete_if_score_lt` removed — the recommendation has been fully flag-combination based since v0.3.x.
- Added `tests/run_tests.py` with checks for the binary, recommendation distribution, config override, JSON validity, CSV format, and CPU/CUDA metric parity.

Output schema, CLI behavior, kernel math, and the COCO val2017 recommendation distribution (4909 keep / 91 review / 0 delete) are unchanged.

### v0.3.1 — CUDA timed-region cleanup

- Reduction scratch buffers (`d_partial` for brightness, contrast, sobel, saturation) are now owned by `CudaQualityAnalyzer` and reused across images — kernel launchers no longer call `cudaMalloc` / `cudaFree` per launch.
- The redundant `cudaDeviceSynchronize()` call inside each kernel launcher was removed. Kernel launches are checked with `cudaGetLastError()` instead. The trailing `cudaEventSynchronize(ev_stop_)` in `compute()` provides the only host-side barrier; the per-launcher D2H `cudaMemcpy` already drains the GPU implicitly when needed.
- `cudaEvent_t` handles for the timed region are now created once in the analyzer constructor and destroyed in the destructor instead of being recreated per image.

Measured on COCO val2017 (5,000 images), CUDA total time drops from **2762 ms (v0.2)** to **1469 ms (v0.3.1)**, with the speedup ratio rising from **2.91×** to **5.99×**. Recommendation distribution is bit-identical: 4,909 keep / 91 review / 0 delete. Output format and recommendation logic are unchanged.

### v0.3 — CUDA memory reuse

- CUDA device memory is now managed through a small RAII wrapper (`CudaBuffer`) instead of raw `cudaMalloc` / `cudaFree`.
- A persistent `CudaQualityAnalyzer` object owns the BGR and grayscale device buffers and reuses them across images, growing them only when a larger image is encountered.
- This removes the per-image `cudaMalloc` / `cudaFree` calls for the two largest device buffers and reduces driver overhead on long scans.
- Cosmetic: the in-place progress line now clears properly between images, and the tool falls back to one line per image when stdout is redirected (so `tee`/log files stay readable).

v0.3 introduced persistent BGR and grayscale device buffers. The larger timed-region improvement landed in v0.3.1, where scratch buffers were reused and redundant synchronizations were removed.

## Limitations

- JPEG/PNG decoding is still done on CPU through OpenCV.
- CUDA scratch buffers are now reused across images, but the current pipeline still processes one image at a time.
- No batch processing yet — each image is processed in its own kernel-launch sequence.
- Blur and detail detection is based on average Sobel edge magnitude — fast, but coarser than Laplacian-of-Gaussian or frequency-domain methods.
- Thresholds are heuristic and should be tuned per-domain (medical imaging, astronomy, low-light surveillance, etc. all behave differently).
- This tool is for **dataset triage**, not final automatic deletion. Always review the `delete` bucket before acting on it.

## Roadmap

### Accuracy & Metrics
- [ ] Add entropy score using grayscale histogram
- [x] Add configurable threshold presets: `relaxed`, `default`, `strict`
- [ ] Add validation on low-light and blur-specific datasets

### Performance
- [x] Reuse CUDA memory buffers across images *(v0.3)*
- [x] Reuse partial-sum / scratch buffers across kernel launches *(v0.3.1)*
- [x] Cache `cudaEvent_t` handles in the analyzer *(v0.3.1)*
- [ ] Add batch processing for multiple images
- [ ] Add pinned memory and asynchronous CUDA transfers
- [ ] Move per-launcher partial-sum reductions onto the GPU to avoid the small D2H copies inside the timed region

### Code Quality
- [x] Add RAII-based CUDA buffer management *(v0.3)*
- [ ] Refactor the C++ orchestration layer into clearer components:
  - `DatasetScanner`
  - `CpuQualityAnalyzer`
  - `CudaQualityAnalyzer`
  - `ReportWriter`
- [ ] Replace manual JSON writing with `nlohmann/json`
- [ ] Consider an optional `stb_image` backend for lighter image loading

### Dataset Management
- [ ] Add duplicate detection via perceptual hash
- [x] Add safe dataset organize mode: `keep/`, `review/`, `delete/`
- [x] Add dry-run mode before moving files

### Integrations
- [ ] Add Python API bindings
- [ ] Add Jetson live camera mode
- [ ] Add Jetson benchmark results

## License

MIT License. See [LICENSE](LICENSE) for details.
