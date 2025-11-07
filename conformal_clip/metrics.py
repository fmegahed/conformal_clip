from __future__ import annotations
from typing import List, Sequence, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def make_true_labels_from_counts(labels: Sequence[str], label_counts: Sequence[int]) -> List[str]:
    """
    Expand label counts into a list of true labels in order.

    Args:
        labels: Sequence of class label strings.
        label_counts: Number of samples for each class, in the same order as labels.

    Returns:
        List of label strings with each label repeated according to its count.

    Example:
        >>> make_true_labels_from_counts(["Nominal", "Defective"], [3, 2])
        ['Nominal', 'Nominal', 'Nominal', 'Defective', 'Defective']
    """
    out: List[str] = []
    for lab, cnt in zip(labels, label_counts):
        out.extend([lab] * int(cnt))
    return out

def _save_cm(cm, labels, path, fname, cm_title):
    """
    Internal helper to save a confusion matrix plot.

    Args:
        cm: Confusion matrix array.
        labels: Class labels for axis tick labels.
        path: Directory to save the plot (defaults to current directory if None).
        fname: Filename for the saved plot.
        cm_title: Title for the confusion matrix plot.
    """
    os.makedirs(path or ".", exist_ok=True)
    fp = os.path.join(path or ".", fname)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(cm_title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(fp, bbox_inches="tight")
    plt.close()


def compute_classification_metrics(
    csv_file: str,
    labels: Sequence[str],
    label_counts: Sequence[int],
    save_confusion_matrix: bool = True,
    cm_file_path: str | None = None,
    cm_file_name: str = "confusion_matrix.png",
    cm_title: str = "Confusion Matrix for Experiment"
) -> pd.DataFrame:
    """Compute standard point-prediction metrics from a results CSV.

    Expects a column named point_prediction. Falls back to classification_result if missing.
    """
    df = pd.read_csv(csv_file)
    n = df.shape[0]
    if len(labels) != len(label_counts) or sum(label_counts) != n:
        raise ValueError("Label counts must match labels and sum to the number of rows")

    y_true = make_true_labels_from_counts(labels, label_counts)
    y_pred = (df["point_prediction"].tolist()
              if "point_prediction" in df.columns
              else df["classification_result"].tolist())

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))

    if len(labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        pos = labels[1]
        precision = precision_score(y_true, y_pred, pos_label=pos, average="binary", zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=pos, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=pos, average="binary", zero_division=0)
        if f"{pos}_prob" in df.columns:
            auc = roc_auc_score([1 if t == pos else 0 for t in y_true], df[f"{pos}_prob"].tolist())
        else:
            auc = float("nan")
    else:
        specificity = float("nan")
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        auc = float("nan")

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Sensitivity (Recall)", "Specificity", "Precision", "F1 Score", "AUC"],
        "Value": [
            accuracy_score(y_true, y_pred),
            recall,
            specificity,
            precision,
            f1,
            auc
        ]
    })

    if save_confusion_matrix:
        _save_cm(cm, labels, cm_file_path, cm_file_name, cm_title)

    return metrics

def compute_conformal_set_metrics(
    csv_file: str,
    labels: Sequence[str],
    label_counts: Sequence[int]
) -> pd.DataFrame:
    """Compute conformal prediction-set metrics from a CSV produced by few_shot_fault_classification_conformal.

    Expects a prediction_set column with either "ABSTAIN" or a pipe-delimited list like "Nominal|Defective".
    Optionally uses alpha and mondrian columns for context.
    """
    df = pd.read_csv(csv_file)
    n = df.shape[0]
    if len(labels) != len(label_counts) or sum(label_counts) != n:
        raise ValueError("Label counts must match labels and sum to the number of rows")

    y_true = make_true_labels_from_counts(labels, label_counts)

    def parse_set(s: str) -> list[str]:
        if isinstance(s, str) and s.strip().upper() == "ABSTAIN":
            return []
        if isinstance(s, str) and len(s.strip()) > 0:
            return [t.strip() for t in s.split("|")]
        return []

    sets = df["prediction_set"].apply(parse_set).tolist()
    sizes = [len(s) for s in sets]
    contains_true = [t in s for t, s in zip(y_true, sets)]

    coverage = float(np.mean(contains_true)) if len(contains_true) else float("nan")

    per_class_cov = {}
    for c in labels:
        idx = [i for i, t in enumerate(y_true) if t == c]
        per_class_cov[c] = float(np.mean([contains_true[i] for i in idx])) if idx else float("nan")

    rows = []
    if "alpha" in df.columns:
        rows.append({"Metric": "Alpha", "Value": df["alpha"].iloc[0]})
    if "mondrian" in df.columns:
        rows.append({"Metric": "Mondrian", "Value": bool(df["mondrian"].iloc[0])})

    rows.extend([
        {"Metric": "Coverage (overall)", "Value": coverage},
        {"Metric": f"Coverage [{labels[0]}]", "Value": per_class_cov[labels[0]]},
        {"Metric": f"Coverage [{labels[1]}]", "Value": per_class_cov[labels[1]]},
        # {"Metric": "Average set size", "Value": float(np.mean(sizes)) if sizes else float("nan")},
        # {"Metric": "Singleton rate", "Value": float(np.mean([s == 1 for s in sizes])) if sizes else float("nan")},
        # {"Metric": "Doubleton rate", "Value": float(np.mean([s == 2 for s in sizes])) if sizes else float("nan")},
        # {"Metric": "Empty set rate", "Value": float(np.mean([s == 0 for s in sizes])) if sizes else float("nan")},
    ])
    return pd.DataFrame(rows)
