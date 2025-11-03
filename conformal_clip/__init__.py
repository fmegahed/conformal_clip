"""
conformal_clip
==================
Utilities to run zero-shot and few-shot CLIP classification with optional
conformal prediction and simple reporting and visualization helpers.
"""
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
