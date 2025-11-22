"""
Few-Shot Textile Classification with MobileNetV4
=================================================

This example demonstrates few-shot conformal prediction on the textile defect dataset
using the MobileNetV4 backend, a vision-only (timm-based) model optimized for efficiency.

Unlike CLIP-like models, MobileNetV4 is a vision-only encoder without text capabilities,
making it suitable for pure image similarity-based classification. It's highly efficient
for CPU and low-resource GPU environments.

The script:
1. Loads the MobileNetV4 model (lightweight CNN from timm)
2. Builds train/calibration/test splits from the textile dataset
3. Runs conformal prediction with Mondrian (class-conditional) thresholds
4. Prints results for each test image

Requirements:
    pip install "conformal-clip[data]"

Usage:
    python textile_mobilenetv4.py

Note: This is a vision-only backend, so zero-shot classification (which requires
      text encoding) is not supported. Only few-shot classification is available.
"""

import os
import torch
from conformal_clip import load_backend, few_shot_fault_classification_conformal
from _shared_textile import build_textile_splits, load_pil


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "mobilenetv4"
    model, preprocess_fn, _ = load_backend(backend, None, device)

    tr_nom, tr_def, calib, calib_labels, test, _ = build_textile_splits()

    prep = lambda paths: [preprocess_fn(img) for img in load_pil(paths)]
    nominal_images = prep(tr_nom)
    defective_images = prep(tr_def)
    calib_images = prep(calib)
    test_images = prep(test)
    test_fnames = [os.path.basename(p) for p in test]

    results = few_shot_fault_classification_conformal(
        model=model,
        test_images=test_images,
        test_image_filenames=test_fnames,
        nominal_images=nominal_images,
        nominal_descriptions=["nom"] * len(nominal_images),
        defective_images=defective_images,
        defective_descriptions=["def"] * len(defective_images),
        calib_images=calib_images,
        calib_labels=calib_labels,
        alpha=0.1,
        temperature=1.0,
        mondrian=True,
        prob_calibration=None,
        allow_empty=False,
        seed=2025,
    )
    print(f"Ran {backend} with {len(results)} test images.")


if __name__ == "__main__":
    main()

