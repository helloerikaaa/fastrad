"""
Cohort-level throughput benchmark: fastrad (GPU, sequential) vs.
PyRadiomics (CPU, multiprocessing) across a batch of independent cases.

*** THIS SCRIPT WAS WRITTEN WITHOUT ACCESS TO A GPU OR REAL TCIA DATA AND
    HAS NEVER BEEN RUN. It is a scaffold, not a verified benchmark. ***

Addresses the reviewer criticism (raised by all four reviewers) that the
paper's single-case latency benchmark (0.116 s/scan -> ~517 scans/min,
corrected from an arithmetic error that previously said 860) does not
demonstrate realistic cohort-level throughput, since PyRadiomics can be
trivially parallelised across independent CPU processes for batch
workloads even though it shows no internal multi-threading benefit
(Section 3.4 of the manuscript).

This script times:
  1. fastrad, sequential single-GPU extraction over N cases.
  2. PyRadiomics, single-process sequential extraction over N cases
     (the naive baseline).
  3. PyRadiomics, multiprocessing.Pool-parallelised extraction over N
     cases across all available CPU cores (the realistic cohort baseline
     that a practitioner would actually use).

Expected data layout: reuses the same multi-case directory structure as
run_multicase_validation.py:
    tests/fixtures/tcia/multicase/<case_id>/image.nrrd
    tests/fixtures/tcia/multicase/<case_id>/label.nrrd

If you only have the single LUNG1-001 case, you can still run this script
against N synthetic copies (see SYNTHETIC_COHORT_SIZE below) to at least
validate the throughput *mechanics* end-to-end, but that is NOT a
substitute for real multi-case cohort data in the final manuscript number
-- it would just be re-extracting the same case N times.
"""
import csv
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from radiomics import featureextractor

from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage

# If no real multi-case directory is populated, fall back to N synthetic
# re-extractions of the single LUNG1-001 case, purely to validate the
# throughput-measurement mechanics. Set to 0 to disable this fallback and
# require real multi-case data instead.
SYNTHETIC_COHORT_SIZE = 20


def _pyrad_worker(args: tuple[str, str, str]) -> float:
    """Runs in a worker process: extract one case with PyRadiomics, return elapsed seconds."""
    img_path, mask_path, config_path = args
    sitk_image = sitk.ReadImage(img_path)
    sitk_mask = sitk.ReadImage(mask_path)
    extractor = featureextractor.RadiomicsFeatureExtractor(config_path)
    t0 = time.perf_counter()
    extractor.execute(sitk_image, sitk_mask)
    return time.perf_counter() - t0


def discover_cases(project_root: Path) -> list[tuple[str, Path, Path]]:
    multicase_dir = project_root / "tests" / "fixtures" / "tcia" / "multicase"
    cases = []
    if multicase_dir.exists():
        for case_dir in sorted(multicase_dir.iterdir()):
            img_path = case_dir / "image.nrrd"
            mask_path = case_dir / "label.nrrd"
            if img_path.exists() and mask_path.exists():
                cases.append((case_dir.name, img_path, mask_path))

    if not cases and SYNTHETIC_COHORT_SIZE > 0:
        single_img = project_root / "tests" / "fixtures" / "tcia" / "lung1_image.nrrd"
        single_mask = project_root / "tests" / "fixtures" / "tcia" / "lung1_label.nrrd"
        if single_img.exists() and single_mask.exists():
            print(f"WARNING: no real multi-case directory found. Falling back to "
                  f"{SYNTHETIC_COHORT_SIZE} synthetic re-extractions of the single "
                  f"LUNG1-001 case. This validates timing mechanics ONLY -- it is "
                  f"NOT a substitute for a real multi-case cohort in the manuscript.")
            cases = [(f"LUNG1-001-rep{i}", single_img, single_mask)
                     for i in range(SYNTHETIC_COHORT_SIZE)]

    return cases


def run():
    print("Running Cohort-Level Throughput Benchmark...")
    project_root = Path(__file__).parent.parent
    config_path = project_root / "pyradiomics_config.yaml"
    output_dir = project_root / "benchmarks" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(project_root)
    if not cases:
        msg = ("Error: no cases found (real multi-case directory is empty and "
               "the single-case fallback file is missing). Cannot run cohort "
               "throughput benchmark.")
        print(msg)
        return msg

    n_cases = len(cases)
    print(f"  -> Cohort size: {n_cases} cases")

    fastrad_settings = FeatureSettings.from_yaml(config_path, device="cuda")
    fastrad_ext = FeatureExtractor(fastrad_settings)

    # 1. fastrad, sequential, single GPU
    print("  -> Timing fastrad sequential GPU extraction...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _case_id, img_path, mask_path in cases:
        sitk_image = sitk.ReadImage(str(img_path))
        sitk_mask = sitk.ReadImage(str(mask_path))
        image_tensor = torch.from_numpy(sitk.GetArrayFromImage(sitk_image)).float()
        mask_tensor = torch.from_numpy(sitk.GetArrayFromImage(sitk_mask)).float()
        spacing = sitk_image.GetSpacing()[::-1]
        f_img = MedicalImage(image_tensor, spacing=spacing)
        f_mask = Mask(mask_tensor, spacing=spacing)
        fastrad_ext.extract(f_img, f_mask)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fastrad_total_time = time.perf_counter() - t0

    # 2. PyRadiomics, sequential, single process
    print("  -> Timing PyRadiomics sequential (1 process) extraction...")
    pyrad_args = [(str(img_path), str(mask_path), str(config_path)) for _cid, img_path, mask_path in cases]
    t0 = time.perf_counter()
    pyrad_seq_times = [_pyrad_worker(a) for a in pyrad_args]
    pyrad_seq_total_time = time.perf_counter() - t0

    # 3. PyRadiomics, multiprocessing across all available CPU cores
    n_workers = mp.cpu_count() or 4
    print(f"  -> Timing PyRadiomics multiprocessing ({n_workers} workers)...")
    t0 = time.perf_counter()
    with mp.Pool(processes=n_workers) as pool:
        pyrad_mp_times = pool.map(_pyrad_worker, pyrad_args)
    pyrad_mp_total_time = time.perf_counter() - t0

    fastrad_throughput = n_cases / fastrad_total_time * 60.0
    pyrad_seq_throughput = n_cases / pyrad_seq_total_time * 60.0
    pyrad_mp_throughput = n_cases / pyrad_mp_total_time * 60.0

    records = [{
        "n_cases": n_cases,
        "n_cpu_workers": n_workers,
        "fastrad_gpu_total_s": fastrad_total_time,
        "fastrad_gpu_scans_per_min": fastrad_throughput,
        "pyradiomics_sequential_total_s": pyrad_seq_total_time,
        "pyradiomics_sequential_scans_per_min": pyrad_seq_throughput,
        "pyradiomics_multiprocess_total_s": pyrad_mp_total_time,
        "pyradiomics_multiprocess_scans_per_min": pyrad_mp_throughput,
        "fastrad_vs_pyrad_multiprocess_speedup": pyrad_mp_total_time / fastrad_total_time,
    }]

    csv_path = output_dir / "cohort_throughput.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary = (
        f"Cohort of {n_cases} cases: fastrad GPU {fastrad_throughput:.1f} scans/min, "
        f"PyRadiomics sequential {pyrad_seq_throughput:.1f} scans/min, "
        f"PyRadiomics {n_workers}-process {pyrad_mp_throughput:.1f} scans/min. "
        f"fastrad vs. realistic (multiprocess) PyRadiomics baseline: "
        f"{pyrad_mp_total_time / fastrad_total_time:.2f}x. See {csv_path}."
    )
    print(summary)
    return summary


if __name__ == "__main__":
    print(run())
