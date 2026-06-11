#!/usr/bin/env python3
"""Lightweight validation tests for CudaDatasetCleaner.

Tests:
  - binary_exists
  - recommendation_distribution (on test_dataset)
  - config_override (custom threshold takes effect)
  - json_validity (output parses cleanly)
  - csv_format (header order + parseable rows)
  - cpu_cuda_parity (metrics within tolerance; skipped if CUDA unavailable)

Usage:
    python3 tests/run_tests.py
"""

import csv
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(ROOT, "build", "cuda-dataset-cleaner")
TEST_DATASET = os.path.join(ROOT, "test_dataset")

EXPECTED_CSV_HEADER = [
    "file_path", "width", "height",
    "brightness", "contrast", "edge_score",
    "saturated_pixel_ratio", "quality_score",
    "flags", "recommendation",
    "cpu_time_ms", "cuda_time_ms",
]

EXPECTED_RECOMMENDATIONS = {
    "sharp_checker.bmp":   "keep",
    "bright.bmp":          "delete",
    "low_contrast.bmp":    "review",
    "smooth_gradient.bmp": "review",
    "color_gradient.bmp":  "review",
    "dark.bmp":            "review",
}


def run_scan(extra_args):
    args = [BINARY, "scan", TEST_DATASET] + list(extra_args)
    return subprocess.run(args, capture_output=True, text=True, timeout=120)


def _basename_map(images):
    return {os.path.basename(img["file_path"]): img for img in images if not img.get("error")}


# --- tests --------------------------------------------------------------

def test_binary_exists():
    if not os.path.isfile(BINARY):
        return False, f"{BINARY} not found - run `cmake --build build` first"
    if not os.access(BINARY, os.X_OK):
        return False, f"{BINARY} is not executable"
    return True, ""


def test_recommendation_distribution():
    with tempfile.TemporaryDirectory() as tmp:
        jp = os.path.join(tmp, "r.json")
        cp = os.path.join(tmp, "s.csv")
        res = run_scan(["--output", jp, "--csv", cp])
        if res.returncode != 0:
            return False, f"binary failed (rc={res.returncode}): {res.stderr.strip()}"
        with open(jp) as f:
            data = json.load(f)
        by_name = _basename_map(data["images"])
        for name, expected in EXPECTED_RECOMMENDATIONS.items():
            actual = by_name.get(name, {}).get("recommendation")
            if actual != expected:
                return False, f"{name}: expected {expected!r}, got {actual!r}"
    return True, ""


def test_config_override():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "cfg.json")
        # An aggressively high low-contrast threshold should flag every image.
        with open(cfg, "w") as f:
            f.write('{"low_contrast_if_less_than": 999.0}')
        jp = os.path.join(tmp, "r.json")
        cp = os.path.join(tmp, "s.csv")
        res = run_scan(["--output", jp, "--csv", cp, "--config", cfg])
        if res.returncode != 0:
            return False, f"binary failed: {res.stderr.strip()}"
        with open(jp) as f:
            data = json.load(f)
        for img in data["images"]:
            if img.get("error"):
                continue
            if "low_contrast" not in img.get("flags", []):
                return False, f"{img['file_path']}: low_contrast missing despite override"
    return True, ""


def test_json_validity():
    with tempfile.TemporaryDirectory() as tmp:
        jp = os.path.join(tmp, "r.json")
        cp = os.path.join(tmp, "s.csv")
        res = run_scan(["--output", jp, "--csv", cp])
        if res.returncode != 0:
            return False, f"binary failed: {res.stderr.strip()}"
        try:
            with open(jp) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return False, f"JSON did not parse: {e}"
        if "summary" not in data or "images" not in data:
            return False, "JSON missing top-level 'summary' or 'images'"
        s = data["summary"]
        if s["total_images"] <= 0:
            return False, "summary.total_images is not positive"
    return True, ""


def test_csv_format():
    with tempfile.TemporaryDirectory() as tmp:
        jp = os.path.join(tmp, "r.json")
        cp = os.path.join(tmp, "s.csv")
        res = run_scan(["--output", jp, "--csv", cp])
        if res.returncode != 0:
            return False, f"binary failed: {res.stderr.strip()}"
        with open(cp, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header != EXPECTED_CSV_HEADER:
                return False, f"header mismatch:\n  got      {header}\n  expected {EXPECTED_CSV_HEADER}"
            rows = list(reader)
            if not rows:
                return False, "no data rows"
            for row in rows:
                if len(row) != len(EXPECTED_CSV_HEADER):
                    return False, f"row width mismatch: {row}"
    return True, ""


def test_preset_selection():
    """Two distinct presets must produce different flag-count distributions.

    Compares strict vs relaxed (rather than default vs preset) because the
    synthetic test_dataset uses extreme metric values that fall outside any
    threshold band default-vs-strict can distinguish.
    """
    cfg = os.path.join(ROOT, "config.json")
    if not os.path.isfile(cfg):
        return "skip", "config.json missing in project root"
    with tempfile.TemporaryDirectory() as tmp:
        s_j = os.path.join(tmp, "strict.json");  s_c = os.path.join(tmp, "strict.csv")
        r_j = os.path.join(tmp, "relaxed.json"); r_c = os.path.join(tmp, "relaxed.csv")
        rs = run_scan(["--output", s_j, "--csv", s_c, "--config", cfg, "--preset", "strict"])
        rr = run_scan(["--output", r_j, "--csv", r_c, "--config", cfg, "--preset", "relaxed"])
        if rs.returncode != 0:
            return False, f"strict scan failed: {rs.stderr.strip()}"
        if rr.returncode != 0:
            return False, f"relaxed scan failed: {rr.stderr.strip()}"
        with open(s_j) as f: s_sum = json.load(f)["summary"]
        with open(r_j) as f: r_sum = json.load(f)["summary"]
        counts = ["blurry_count", "too_dark_count", "low_contrast_count"]
        if all(s_sum[k] == r_sum[k] for k in counts):
            return False, "strict and relaxed presets produced identical flag counts"
    return True, ""


def test_organize_dry_run():
    """Without --copy-files, --organize-output writes plan only — no buckets."""
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "organize_out")
        jp = os.path.join(tmp, "r.json"); cp = os.path.join(tmp, "s.csv")
        res = run_scan(["--output", jp, "--csv", cp, "--organize-output", out_dir])
        if res.returncode != 0:
            return False, f"dry-run scan failed: {res.stderr.strip()}"
        plan = os.path.join(out_dir, "organize_plan.csv")
        if not os.path.isfile(plan):
            return False, "organize_plan.csv was not created"
        for bucket in ("keep", "review", "delete"):
            if os.path.isdir(os.path.join(out_dir, bucket)):
                return False, f"dry-run created {bucket}/ subdir (should not copy)"
        with open(plan, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header != ["source_path", "destination_path", "recommendation", "action"]:
                return False, f"plan header mismatch: {header}"
            rows = list(reader)
            if not rows:
                return False, "plan has no rows"
            for row in rows:
                if row[3] != "dry-run-copy" and row[3] != "skip":
                    return False, f"unexpected action in dry-run plan: {row}"
    return True, ""


def test_organize_copy():
    """With --copy-files, files land in keep/review/delete buckets; originals stay."""
    # Snapshot source dataset modification times before the run.
    src_files = []
    for root, _, files in os.walk(TEST_DATASET):
        for fn in files:
            p = os.path.join(root, fn)
            src_files.append((p, os.path.getmtime(p)))
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "organize_out")
        jp = os.path.join(tmp, "r.json"); cp = os.path.join(tmp, "s.csv")
        res = run_scan(["--output", jp, "--csv", cp,
                        "--organize-output", out_dir, "--copy-files"])
        if res.returncode != 0:
            return False, f"copy scan failed: {res.stderr.strip()}"
        copied = 0
        for root, _, files in os.walk(out_dir):
            for fn in files:
                if fn != "organize_plan.csv":
                    copied += 1
        if copied == 0:
            return False, "no files were copied"
    # Verify nothing under the source dataset was modified
    for p, mtime_before in src_files:
        if not os.path.exists(p):
            return False, f"source file deleted by organize: {p}"
        if abs(os.path.getmtime(p) - mtime_before) > 1e-3:
            return False, f"source file mtime changed: {p}"
    return True, ""


def test_overlap_refusal():
    """Overlapping input/output must refuse with non-zero exit."""
    with tempfile.TemporaryDirectory() as tmp:
        jp = os.path.join(tmp, "r.json"); cp = os.path.join(tmp, "s.csv")
        res = run_scan(["--output", jp, "--csv", cp,
                        "--organize-output", TEST_DATASET, "--copy-files"])
        if res.returncode == 0:
            return False, "binary did not refuse overlapping output dir"
        if "overlap" not in res.stderr.lower() and "refus" not in res.stderr.lower():
            return False, f"refusal message unclear: {res.stderr.strip()}"
    return True, ""


def _run_binary_raw(args):
    """Run the binary without inserting `scan <TEST_DATASET>`; used for help/version/CLI checks."""
    cmd = [BINARY] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


def test_help_flag():
    """--help prints usage and exits 0; mentions key options."""
    res = _run_binary_raw(["--help"])
    if res.returncode != 0:
        return False, f"--help returned non-zero exit ({res.returncode})"
    out = res.stdout
    for needle in ("--organize-output", "--preset", "--copy-files", "scan"):
        if needle not in out:
            return False, f"--help output missing '{needle}'"
    return True, ""


def test_version_flag():
    """--version prints version line containing v0.5 and exits 0."""
    res = _run_binary_raw(["--version"])
    if res.returncode != 0:
        return False, f"--version returned non-zero exit ({res.returncode})"
    if "v0.5" not in res.stdout:
        return False, f"--version output missing 'v0.5': {res.stdout!r}"
    # The bare `version` command must work too.
    res2 = _run_binary_raw(["version"])
    if res2.returncode != 0 or "v0.5" not in res2.stdout:
        return False, f"'version' command did not match: rc={res2.returncode} out={res2.stdout!r}"
    return True, ""


def test_invalid_flag_exit_nonzero():
    """An unknown scan flag exits non-zero with a hint pointing at --help."""
    res = run_scan(["--this-flag-does-not-exist"])
    if res.returncode == 0:
        return False, "unknown flag did not produce non-zero exit"
    if "--help" not in res.stderr:
        return False, f"unknown-flag error should hint about --help: {res.stderr!r}"
    return True, ""


def test_copy_files_requires_organize_output():
    """--copy-files without --organize-output exits non-zero."""
    res = run_scan(["--copy-files"])
    if res.returncode == 0:
        return False, "--copy-files without --organize-output should fail"
    if "organize-output" not in res.stderr:
        return False, f"error should mention --organize-output: {res.stderr!r}"
    return True, ""


def test_no_cpu_no_cuda_rejected():
    """Disabling both metric backends is rejected at CLI validation."""
    res = run_scan(["--no-cpu", "--no-cuda"])
    if res.returncode == 0:
        return False, "--no-cpu --no-cuda should fail"
    return True, ""


def test_cpu_cuda_parity():
    with tempfile.TemporaryDirectory() as tmp:
        # CPU-only
        cpu_j = os.path.join(tmp, "cpu.json")
        cpu_c = os.path.join(tmp, "cpu.csv")
        rc = run_scan(["--output", cpu_j, "--csv", cpu_c, "--no-cuda"])
        if rc.returncode != 0:
            return False, f"CPU-only scan failed: {rc.stderr.strip()}"

        # CUDA-only
        cuda_j = os.path.join(tmp, "cuda.json")
        cuda_c = os.path.join(tmp, "cuda.csv")
        rg = run_scan(["--output", cuda_j, "--csv", cuda_c, "--no-cpu"])
        if rg.returncode != 0:
            return "skip", "CUDA-only scan failed (no GPU?)"

        with open(cpu_j) as f:
            cpu_imgs = _basename_map(json.load(f)["images"])
        with open(cuda_j) as f:
            cuda_imgs = _basename_map(json.load(f)["images"])

        if not cuda_imgs:
            return "skip", "CUDA produced no metrics (no GPU?)"

        tolerances = {
            "brightness":            0.5,
            "contrast":              0.5,
            "edge_score":            2.0,
            "saturated_pixel_ratio": 0.005,
        }
        for name, cpu_img in cpu_imgs.items():
            cuda_img = cuda_imgs.get(name)
            if cuda_img is None:
                continue
            for metric, tol in tolerances.items():
                a = float(cpu_img.get(metric, 0.0))
                b = float(cuda_img.get(metric, 0.0))
                if abs(a - b) > tol:
                    return False, (
                        f"{name}: {metric} CPU={a:.4f} vs CUDA={b:.4f} "
                        f"(|diff|={abs(a-b):.4f} > tol {tol})"
                    )
    return True, ""


# --- driver -------------------------------------------------------------

TESTS = [
    ("binary_exists",                    test_binary_exists),
    ("recommendation_distribution",      test_recommendation_distribution),
    ("config_override",                  test_config_override),
    ("json_validity",                    test_json_validity),
    ("csv_format",                       test_csv_format),
    ("preset_selection",                 test_preset_selection),
    ("organize_dry_run",                 test_organize_dry_run),
    ("organize_copy",                    test_organize_copy),
    ("overlap_refusal",                  test_overlap_refusal),
    ("help_flag",                        test_help_flag),
    ("version_flag",                     test_version_flag),
    ("invalid_flag_exit_nonzero",        test_invalid_flag_exit_nonzero),
    ("copy_files_requires_organize",     test_copy_files_requires_organize_output),
    ("no_cpu_no_cuda_rejected",          test_no_cpu_no_cuda_rejected),
    ("cpu_cuda_parity",                  test_cpu_cuda_parity),
]


def main():
    if not os.path.isdir(TEST_DATASET):
        print(f"ERROR: test_dataset directory not found at {TEST_DATASET}")
        sys.exit(2)

    passed = 0
    failed = 0
    skipped = 0
    for name, fn in TESTS:
        try:
            result = fn()
        except Exception as e:
            print(f"ERROR {name}: {e}")
            failed += 1
            continue

        if isinstance(result, tuple) and result and result[0] == "skip":
            print(f"SKIP {name}: {result[1]}")
            skipped += 1
        elif isinstance(result, tuple) and result[0] is True:
            print(f"PASS {name}")
            passed += 1
        else:
            msg = result[1] if isinstance(result, tuple) and len(result) > 1 else ""
            print(f"FAIL {name}: {msg}")
            failed += 1

    total = len(TESTS)
    print()
    print(f"{passed} passed, {failed} failed, {skipped} skipped (of {total})")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
