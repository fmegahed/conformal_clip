"""
Visualization
=============
Utilities for plotting confusion matrices and other classification visualizations.

This module provides simple plotting functions for confusion matrices
with customizable styling and optional saving to disk.
"""

from __future__ import annotations
from typing import Sequence
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, labels: Sequence[str], title: str = "Confusion Matrix", save_path: str | None = None):
    """
    Plot and optionally save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array (typically from sklearn.metrics.confusion_matrix).
        labels: Class labels for axis tick labels.
        title: Plot title (default "Confusion Matrix").
        save_path: If provided, saves the figure to this path; otherwise displays it.

    Example:
        >>> from sklearn.metrics import confusion_matrix
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> plot_confusion_matrix(cm, labels=["Nominal", "Defective"],
        ...                       title="My Confusion Matrix",
        ...                       save_path="results/cm.png")
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
