from __future__ import annotations
from typing import List, Optional, Tuple
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_zero_shot_predictions(
    labels: List[str],
    label_counts: List[int],
    test_images: List[torch.Tensor],
    test_image_filenames: List[str],
    model,
    device: torch.device,
    clip_module=None,
    save_confusion_matrix: bool = False,
    cm_title: str = "Confusion Matrix for Zero Shot Classification",
    short_labels: Optional[List[str]] = None,
    cm_file_path: Optional[str] = None,
    cm_file_name: str = "confusion_matrix.png"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate zero-shot predictions using a CLIP-like model and preprocessed image tensors.

    Args:
        labels: Text strings for classes in order.
        label_counts: Number of test images per label, same order as labels.
        test_images: List of preprocessed image tensors ready for model.encode_image.
        test_image_filenames: Filenames for each test image.
        model: CLIP-like model with encode_text and encode_image.
        device: Torch device used for computations.
        clip_module: The clip module providing tokenize. If None, raises.
        save_confusion_matrix: If True, saves a confusion matrix plot.
        cm_title: Title for confusion matrix.
        short_labels: Optional short class labels for the plot.
        cm_file_path: Directory to save the confusion matrix image.
        cm_file_name: Filename for saved plot.

    Returns:
        metrics_df: Table of global classification metrics.
        results_df: Per-image results including probabilities.
    """
    if clip_module is None:
        raise ValueError("clip_module must be provided to tokenize labels")

    text_inputs = clip_module.tokenize(labels).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    predicted_labels_idx: List[int] = []
    results_data = []

    true_labels_idx = [i for i, cnt in enumerate(label_counts) for _ in range(cnt)]

    for idx, image_tensor in enumerate(test_images):
        with torch.no_grad():
            image_features = model.encode_image(image_tensor.to(device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pred_idx = int(similarity.argmax().item())
            probs = similarity.squeeze().tolist()

            predicted_labels_idx.append(pred_idx)
            results_data.append({
                "image_filename": test_image_filenames[idx],
                "true_label": labels[true_labels_idx[idx]],
                "predicted_label": labels[pred_idx],
                **{f"prob_{lbl}" : prob for lbl, prob in zip(labels, probs)}
            })

    results_df = pd.DataFrame(results_data)

    y_true = [r["true_label"] for r in results_data]
    y_pred = [r["predicted_label"] for r in results_data]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if len(labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        positive_label = labels[1]
        y_true_bin = [1 if t == positive_label else 0 for t in y_true]
        auc = (
            roc_auc_score(y_true_bin, results_df[f"prob_{positive_label}"].tolist())
            if f"prob_{positive_label}" in results_df.columns else float("nan")
        )
        precision = precision_score(y_true, y_pred, pos_label=positive_label, average="binary")
        recall = recall_score(y_true, y_pred, pos_label=positive_label, average="binary")
        f1 = f1_score(y_true, y_pred, pos_label=positive_label, average="binary")
    else:
        specificity = float("nan")
        auc = float("nan")
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy", "Sensitivity (Recall)", "Specificity", "Precision", "F1 Score", "AUC"
        ],
        "Value": [
            accuracy_score(y_true, y_pred),
            recall,
            specificity,
            precision,
            f1,
            auc,
        ]
    })

    if save_confusion_matrix:
        if short_labels is None:
            short_labels = labels
        os.makedirs(cm_file_path or ".", exist_ok=True)
        save_path = os.path.join(cm_file_path or ".", cm_file_name)
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=short_labels, yticklabels=short_labels)
        plt.title(cm_title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    return metrics_df, results_df
