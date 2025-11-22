"""
Systematic Benchmarking Across Backends
========================================

This example demonstrates comprehensive benchmarking across multiple model backends,
calibration methods, and conformal prediction settings on the textile defect dataset.

The script:
1. Builds consistent train/calibration/test splits
2. Evaluates multiple backends (controlled by resource_tier parameter)
3. Tests different calibration methods (none, isotonic, sigmoid/Platt)
4. Compares conformal prediction modes (none, global, Mondrian)
5. Saves results as CSV and styled HTML tables with highlighted best metrics

Requirements:
    pip install "conformal-clip[data]"

Usage:
    python benchmark_microstructure.py

Output:
    - results/benchmark_classification.csv: Point prediction metrics
    - results/benchmark_conformal.csv: Conformal set metrics
    - results/benchmark_classification.html: Styled classification table
    - results/benchmark_conformal.html: Styled conformal metrics table

Note: By default, resource_tier="low" runs only small models suitable for low-resource
      environments. Set to "medium" or "high" to include larger models if your hardware
      supports them. Running the full benchmark can require substantial memory and time.
"""

import os
import torch
import pandas as pd
from conformal_clip import benchmark_models
from _shared_microstructure import build_microstructure_splits, load_pil_from_github
from conformal_clip import load_backend  # only to get preprocess for our dataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build consistent splits for all backends
    tr_nom, tr_def, calib, calib_labels, test, test_labels = build_microstructure_splits()
    # We'll let each backend preprocess internally in benchmark_models; here we keep PILs
    train_nominal_images = load_pil_from_github(tr_nom)
    train_defective_images = load_pil_from_github(tr_def)
    calib_images = load_pil_from_github(calib)
    test_images = load_pil_from_github(test)

    cls_df, cp_df, cls_style, cp_style = benchmark_models(
        train_nominal_images=train_nominal_images,
        train_defective_images=train_defective_images,
        calib_images=calib_images,
        calib_labels=calib_labels,
        test_images=test_images,
        test_labels=test_labels,
        device=device,
        temperature=1.0,
        seed=2025,
        # By default we use the "low" resource tier so the benchmark runs quickly
        # and only loads smaller backbones. Set resource_tier="medium" or "high"
        # to include larger models if your hardware can support them.
        resource_tier="high",
        calibration_methods=(None, "isotonic", "sigmoid"),
        conformal_modes=(None, "global", "mondrian"),
        alpha_list=(0.1,),
        allow_empty=False,
        csv_path="results/microstructure",
        csv_prefix="microstructure_"
    )

    os.makedirs("results", exist_ok=True)
    cls_df.to_csv("results/microstructure_benchmark_classification.csv", index=False)
    cp_df.to_csv("results/microstructure_benchmark_conformal.csv", index=False)
    cls_style.to_html("results/microstructure_benchmark_classification.html")
    cp_style.to_html("results/microstructure_benchmark_conformal.html")
    print("Saved results to results/benchmark_*.{csv,html}")


if __name__ == "__main__":
    main()
