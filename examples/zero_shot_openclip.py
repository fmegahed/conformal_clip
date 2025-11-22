"""
Zero-Shot Textile Classification with OpenCLIP
===============================================

This example demonstrates zero-shot classification using a CLIP-like vision-language
model on the textile defect dataset. Unlike few-shot methods, zero-shot classification
uses only text prompts (no exemplar images) to classify test images.

The script:
1. Loads an OpenCLIP model (openclipbase backend)
2. Builds a test split from the textile dataset
3. Evaluates zero-shot predictions using text labels only
4. Computes and displays classification metrics

Requirements:
    pip install "conformal-clip[data]"

Usage:
    python zero_shot_openclip.py

Note: Zero-shot only works with CLIP-like (vision-language) backends that support
      text encoding. Vision-only backends (e.g., timm models) do not support this mode.
"""

import os
import torch
from PIL import Image
from conformal_clip import load_backend, evaluate_zero_shot_predictions
from _shared_textile import build_textile_splits, load_pil


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "openclipbase"  # any CLIP-like backend key
    model, preprocess_fn, tokenize_fn = load_backend(backend, None, device)

    # Build a test split from the textile dataset
    _, _, _, _, test_paths, test_labels = build_textile_splits()

    # Preprocess PIL images for the model
    pil_images = load_pil(test_paths)
    test_images = [preprocess_fn(img) for img in pil_images]
    test_filenames = [os.path.basename(p) for p in test_paths]

    # Labels and counts for zero-shot evaluation
    labels = ["Nominal", "Defective"]
    label_counts = [sum(1 for t in test_labels if t == labels[0]), sum(1 for t in test_labels if t == labels[1])]

    metrics_df, results_df = evaluate_zero_shot_predictions(
        labels=labels,
        label_counts=label_counts,
        test_images=test_images,
        test_image_filenames=test_filenames,
        model=model,
        device=device,
        tokenize_fn=tokenize_fn,  # from load_backend
        save_confusion_matrix=False,
    )

    print("Zero-shot metrics:\n", metrics_df)
    print("Results head:\n", results_df.head())


if __name__ == "__main__":
    main()

