"""
conformal_clip
==============
Few-shot vision classification with conformal prediction and optional probability calibration.

This package provides:
- Backend loading for OpenCLIP (vision-language) and timm (vision-only) models
- Few-shot classification using exemplar image banks
- Conformal prediction (global and Mondrian) with finite-sample coverage guarantees
- Optional probability calibration (isotonic regression or Platt scaling)
- Zero-shot baseline evaluation for CLIP-like models
- Benchmarking utilities to compare backends and settings

Quick Start
-----------
    >>> import torch
    >>> from conformal_clip import load_backend, few_shot_fault_classification_conformal
    >>>
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> model, preprocess_fn, _ = load_backend("openclipbase", None, device)
    >>>
    >>> # Prepare your images (lists of PIL images)
    >>> nominal_images = [preprocess_fn(img) for img in nominal_pil_images]
    >>> defective_images = [preprocess_fn(img) for img in defective_pil_images]
    >>> calib_images = [preprocess_fn(img) for img in calib_pil_images]
    >>> test_images = [preprocess_fn(img) for img in test_pil_images]
    >>>
    >>> results = few_shot_fault_classification_conformal(
    ...     model=model,
    ...     test_images=test_images,
    ...     test_image_filenames=test_filenames,
    ...     nominal_images=nominal_images,
    ...     nominal_descriptions=["..."] * len(nominal_images),
    ...     defective_images=defective_images,
    ...     defective_descriptions=["..."] * len(defective_images),
    ...     calib_images=calib_images,
    ...     calib_labels=calib_labels,
    ...     alpha=0.1,
    ...     mondrian=True,
    ...     prob_calibration="isotonic"
    ... )
"""

__version__ = "0.2.1"
__author__ = "Fadel M. Megahed, Ying-Ju (Tessa) Chen"
__email__ = "fmegahed@miamioh.edu, ychen4@udayton.edu"

# NOTE: To avoid importing optional HTTP stack (requests/urllib3) or other optional
# dependencies during package import, we expose lightweight wrappers for selected
# utilities that import their modules only when called.

def get_image_urls(*args, **kwargs):
    from .io_github import get_image_urls as _f
    return _f(*args, **kwargs)


def load_image(*args, **kwargs):
    from .image_io import load_image as _f
    return _f(*args, **kwargs)


from .zero_shot import evaluate_zero_shot_predictions
from .wrappers import CLIPWrapper, encode_and_normalize
from .conformal import few_shot_fault_classification_conformal
from .backends import load_backend, VISION_LANGUAGE_BACKENDS, VISION_ONLY_BACKENDS
from .metrics import (
    compute_classification_metrics,
    compute_conformal_set_metrics,
    make_true_labels_from_counts,
)
from .viz import plot_confusion_matrix
from .benchmark import benchmark_models

__all__ = [
    "get_image_urls",
    "load_image",
    "evaluate_zero_shot_predictions",
    "CLIPWrapper",
    "encode_and_normalize",
    "few_shot_fault_classification_conformal",
    "load_backend",
    "VISION_LANGUAGE_BACKENDS",
    "VISION_ONLY_BACKENDS",
    "compute_classification_metrics",
    "compute_conformal_set_metrics",
    "make_true_labels_from_counts",
    "plot_confusion_matrix",
    "benchmark_models",
]
