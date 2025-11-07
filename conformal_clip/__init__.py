"""
conformal_clip
==================
Utilities for zero-shot and few-shot CLIP classification with optional
conformal prediction, reporting, and visualization helpers.
"""

__version__ = "0.1.1"
__author__ = "Fadel M. Megahed, Ying-Ju (Tessa) Chen"
__email__ = "fmegahed@miamioh.edu, ychen4@udayton.edu"

from .io_github import get_image_urls
from .image_io import load_image
from .zero_shot import evaluate_zero_shot_predictions
from .wrappers import CLIPWrapper, encode_and_normalize
from .conformal import few_shot_fault_classification_conformal
from .metrics import (
    compute_classification_metrics,
    compute_conformal_set_metrics,
    make_true_labels_from_counts,
)
from .viz import plot_confusion_matrix

__all__ = [
    "get_image_urls",
    "load_image",
    "evaluate_zero_shot_predictions",
    "CLIPWrapper",
    "encode_and_normalize",
    "few_shot_fault_classification_conformal",
    "compute_classification_metrics",
    "compute_conformal_set_metrics",
    "make_true_labels_from_counts",
    "plot_confusion_matrix",
]
