"""
Conformal Prediction
====================
Few-shot classification with conformal prediction for finite-sample coverage guarantees.

This module implements both global and Mondrian (class-conditional) conformal prediction
with optional probability calibration (isotonic regression or Platt scaling).
"""

from __future__ import annotations
from typing import Dict, Any, Sequence
import os
import csv
from datetime import datetime

import numpy as np
import torch

from .wrappers import CLIPWrapper, encode_and_normalize


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Conservative finite-sample quantile used by conformal prediction.

    Uses ceil((1 - alpha) * (n + 1)) with higher-style selection.
    """
    n = len(scores)
    if n <= 0:
        raise ValueError("scores must be non-empty")
    k = int(np.ceil((1.0 - alpha) * (n + 1)))
    k = min(max(k, 1), n)
    return float(np.partition(scores, k - 1)[k - 1])


def _get_nonconformity_scores(
    estimator,
    calib_X,  # images or features
    calib_labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shared helper to compute s = 1 - p_true on the calibration split and return (s, y, classes_).
    """
    probs = estimator.predict_proba(calib_X)  # [n_cal, C]
    y = np.array(list(calib_labels))
    # Efficient label to index map
    label_to_idx = {label: i for i, label in enumerate(estimator.classes_)}
    idx_true = np.array([label_to_idx[lab] for lab in y])
    p_true = probs[np.arange(len(y)), idx_true]
    s = 1.0 - p_true
    return s, y, estimator.classes_


def _fit_global_threshold(
    estimator,
    calib_X,  # images or features
    calib_labels: Sequence[str],
    alpha: float
) -> float:
    """
    Compute a single global threshold q on scores s = 1 - p_true across all classes.
    """
    s, _, _ = _get_nonconformity_scores(estimator, calib_X, calib_labels)
    return _finite_sample_quantile(s, alpha)


def _fit_mondrian_thresholds(
    estimator,
    calib_X,  # images or features
    calib_labels: Sequence[str],
    alpha: float
) -> Dict[str, float]:
    """
    Compute class-conditional thresholds q_y on scores s = 1 - p_true.
    """
    s, y, classes = _get_nonconformity_scores(estimator, calib_X, calib_labels)

    q_map: Dict[str, float] = {}
    for cls in classes:
        mask = (y == cls)
        if not np.any(mask):
            continue
        q_map[cls] = _finite_sample_quantile(s[mask], alpha)

    # Fallback for classes unseen in calibration
    if len(q_map) < len(classes):
        q_global = _finite_sample_quantile(s, alpha)
        for cls in classes:
            q_map.setdefault(cls, q_global)

    return q_map


def _predict_sets_global(
    estimator,
    X,  # images or features
    q: float,
    allow_empty: bool = False
):
    """
    Predict conformal sets using a single global threshold.

    Args:
        estimator: Fitted classifier with predict_proba method.
        X: Images or precomputed features.
        q: Global nonconformity score threshold.
        allow_empty: If False, forces at least one label (argmax).

    Returns:
        List of prediction sets, one per sample in X.
    """
    probs = estimator.predict_proba(X)  # [n, C]
    sets = []
    for i in range(probs.shape[0]):
        inc = [str(estimator.classes_[j]) for j in range(probs.shape[1]) if probs[i, j] >= 1.0 - q]
        if not allow_empty and len(inc) == 0:
            j_star = int(np.argmax(probs[i]))
            inc = [str(estimator.classes_[j_star])]
        sets.append(inc)
    return sets


def _predict_sets_mondrian(
    estimator,
    X,  # images or features
    q_map: Dict[str, float],
    allow_empty: bool = False
):
    """
    Predict conformal sets using class-conditional (Mondrian) thresholds.

    Args:
        estimator: Fitted classifier with predict_proba method.
        X: Images or precomputed features.
        q_map: Dictionary mapping class labels to their thresholds.
        allow_empty: If False, forces at least one label (argmax).

    Returns:
        List of prediction sets, one per sample in X.
    """
    probs = estimator.predict_proba(X)  # [n, C]
    sets = []
    for i in range(probs.shape[0]):
        inc = []
        for j, cls in enumerate(estimator.classes_):
            q = q_map[cls]
            if probs[i, j] >= 1.0 - q:
                inc.append(str(cls))
        if not allow_empty and len(inc) == 0:
            j_star = int(np.argmax(probs[i]))
            inc = [str(estimator.classes_[j_star])]
        sets.append(inc)
    return sets


def few_shot_fault_classification_conformal(
    model,
    test_images,
    test_image_filenames,
    nominal_images,
    nominal_descriptions,
    defective_images,
    defective_descriptions,
    calib_images,
    calib_labels,
    alpha: float = 0.1,
    temperature: float = 1.0,
    mondrian: bool = True,
    class_labels = ("Nominal", "Defective"),
    csv_path: str | None = None,
    csv_filename: str = "image_classification_results_conformal.csv",
    print_one_liner: bool = False,
    seed: int | None = 2025,
    prob_calibration: str | None = None,   # None | "isotonic" | "sigmoid",
    allow_empty: bool = False,
):
    """
    Few-shot CLIP classification with conformal prediction and optional probability calibration.

    This function performs few-shot image classification using CLIP's vision encoder only (no text encoding
    for test images). It creates set-valued predictions with finite-sample coverage guarantees via conformal
    prediction, and optionally calibrates probabilities using isotonic regression or Platt scaling.

    Images are encoded exactly once and features are reused throughout to maximize efficiency.

    Args:
        model: CLIP model with encode_image method.
        test_images: List of preprocessed test image tensors.
        test_image_filenames: List of filenames corresponding to test_images.
        nominal_images: List of preprocessed nominal (non-defective) exemplar images.
        nominal_descriptions: List of text descriptions for nominal exemplars (used for traceability only).
        defective_images: List of preprocessed defective exemplar images.
        defective_descriptions: List of text descriptions for defective exemplars (used for traceability only).
        calib_images: List of preprocessed calibration image tensors.
        calib_labels: List of ground-truth labels for calibration images (e.g., ["Nominal", "Defective", ...]).
        alpha: Miscoverage level for conformal prediction (default 0.1 targets ~90% coverage).
        temperature: Temperature scaling parameter for softmax (default 1.0).
        mondrian: If True, uses class-conditional (Mondrian) conformal prediction; if False, uses global.
        class_labels: Tuple of class names (default ("Nominal", "Defective")).
        csv_path: Directory to save the output CSV (if None, CSV is not saved).
        csv_filename: Name of the output CSV file (default "image_classification_results_conformal.csv").
        print_one_liner: If True, prints a summary line for each test image.
        seed: Random seed for reproducibility (default 2025; set to None to disable seeding).
        prob_calibration: Probability calibration method: None, "isotonic", or "sigmoid" (default None).
        allow_empty: If True, allows empty prediction sets (abstention); if False, forces at least one label.

    Returns:
        List of dictionaries, one per test image, containing:
            - datetime_of_operation: Timestamp of prediction.
            - alpha: Miscoverage level used.
            - temperature: Temperature parameter used.
            - mondrian: Whether Mondrian conformal was used.
            - image_path: Full path to the test image.
            - image_name: Basename of the test image.
            - point_prediction: Argmax class label.
            - prediction_set: Pipe-delimited string of labels in the conformal set (or "ABSTAIN").
            - set_size: Number of labels in the prediction set (0 for "ABSTAIN").
            - {class_0}_prob: Calibrated probability for class 0.
            - {class_1}_prob: Calibrated probability for class 1.
            - nominal_description: Description of the closest nominal exemplar.
            - defective_description: Description of the closest defective exemplar (or "N/A").

    Notes:
        - The function relies on CLIP's vision encoder only; text prompts for test images are not used.
        - For probability calibration, test both "isotonic" and "sigmoid" on your data. According to sklearn
          documentation, isotonic regression preserves monotonicity but may overfit on small calibration sets,
          while sigmoid (Platt) scaling may be more robust for smaller sets. In our experiments with 100
          calibration samples, isotonic performed better on textile defect images.
        - Mondrian conformal provides per-class coverage; global conformal provides overall coverage.

    Example:
        >>> results = few_shot_fault_classification_conformal(
        ...     model=clip_model,
        ...     test_images=test_imgs,
        ...     test_image_filenames=test_files,
        ...     nominal_images=nom_bank,
        ...     nominal_descriptions=["nominal 1", "nominal 2"],
        ...     defective_images=def_bank,
        ...     defective_descriptions=["defect 1", "defect 2"],
        ...     calib_images=calib_imgs,
        ...     calib_labels=calib_lbls,
        ...     alpha=0.1,
        ...     mondrian=True,
        ...     prob_calibration="isotonic"
        ... )
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) Encode few-shot banks
    nominal_feats = encode_and_normalize(model, list(nominal_images))
    defective_feats = (
        encode_and_normalize(model, list(defective_images))
        if len(list(defective_images)) > 0 else torch.empty(0)
    )

    # 2) Base estimator
    clip_est = CLIPWrapper(
        model=model,
        nominal_feats=nominal_feats,
        defective_feats=defective_feats,
        temperature=temperature,
        class_labels=list(class_labels),
    )

    # Precompute features to avoid repeated forward passes
    calib_feats = encode_and_normalize(model, list(calib_images))
    test_feats  = encode_and_normalize(model, list(test_images))

    # ---- Probability calibration (no CalibratedClassifierCV) -----------------
    class _CalibratedProbaWrapper:
        def __init__(self, base_est, transform_fn, classes_):
            self.base_est = base_est
            self.transform_fn = transform_fn  # maps p_uncal_pos -> p_cal_pos
            self.classes_ = np.array(classes_)
            self._estimator_type = "classifier"

        def predict_proba(self, X):
            # Accept images or features, and delegate to base_est
            p = self.base_est.predict_proba(X)      # [n, 2]
            p_pos_uncal = p[:, 1]
            p_pos = self.transform_fn(p_pos_uncal)
            p_pos = np.clip(p_pos, 1e-8, 1 - 1e-8)
            return np.column_stack([1.0 - p_pos, p_pos])

        def predict(self, X):
            p = self.predict_proba(X)
            idx = np.argmax(p, axis=1)
            return self.classes_[idx]

    estimator = clip_est
    if prob_calibration in {"isotonic", "sigmoid"}:
        # Uncalibrated positive-class probabilities on calibration split (features)
        p_cal_uncal = clip_est.predict_proba(calib_feats)[:, 1]
        y_cal = np.array([1 if lab == class_labels[1] else 0 for lab in calib_labels], dtype=int)

        if prob_calibration == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_cal_uncal, y_cal)

            def transform_fn(p):
                p = np.asarray(p)
                return iso.predict(p)

        else:  # "sigmoid" (Platt via logistic regression on logit)
            from sklearn.linear_model import LogisticRegression
            eps = 1e-6

            def to_logit(q):
                q = np.clip(q, eps, 1 - eps)
                return np.log(q / (1 - q))

            lr = LogisticRegression(solver="lbfgs")
            lr.fit(to_logit(p_cal_uncal).reshape(-1, 1), y_cal)

            def transform_fn(p):
                p = np.asarray(p)
                return lr.predict_proba(to_logit(p).reshape(-1, 1))[:, 1]

        estimator = _CalibratedProbaWrapper(clip_est, transform_fn, class_labels)
    # --------------------------------------------------------------------------

    # 3) Conformal thresholds on calibrated probabilities, using precomputed calib_feats
    if mondrian:
        q_map = _fit_mondrian_thresholds(estimator, calib_feats, list(calib_labels), alpha)
        pred_sets = _predict_sets_mondrian(estimator, test_feats, q_map, allow_empty=allow_empty)
    else:
        q_global = _fit_global_threshold(estimator, calib_feats, list(calib_labels), alpha)
        pred_sets = _predict_sets_global(estimator, test_feats, q_global, allow_empty=allow_empty)

    # 4) Point predictions and probabilities from same estimator, on precomputed test_feats
    y_point = estimator.predict(test_feats)
    proba   = estimator.predict_proba(test_feats)  # [n, 2]

    # 5) Per-image traceability and CSV accumulation
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for idx in range(len(test_feats)):
        tf = test_feats[idx]  # reuse precomputed embedding
        max_nom_idx = int(torch.argmax(tf @ nominal_feats.T).item())
        max_def_idx = int(torch.argmax(tf @ defective_feats.T).item()) if defective_feats.nelement() > 0 else -1

        set_labels = pred_sets[idx]
        set_string = "ABSTAIN" if len(set_labels) == 0 else "|".join(set_labels)

        row = {
            "datetime_of_operation": datetime.now().isoformat(),
            "alpha": alpha,
            "temperature": temperature,
            "mondrian": bool(mondrian),
            "image_path": test_image_filenames[idx],
            "image_name": os.path.basename(str(test_image_filenames[idx])),
            "point_prediction": str(y_point[idx]),
            "prediction_set": set_string,
            "set_size": 0 if set_string == "ABSTAIN" else len(set_labels),
            f"{class_labels[0]}_prob": round(float(proba[idx, 0]), 3),
            f"{class_labels[1]}_prob": round(float(proba[idx, 1]), 3),
            "nominal_description": (
                nominal_descriptions[max_nom_idx] if len(nominal_descriptions) > 0 else ""
            ),
            "defective_description": (
                defective_descriptions[max_def_idx] if max_def_idx >= 0 and len(defective_descriptions) > 0 else "N/A"
            ),
        }
        rows.append(row)
        results.append(row)

        if print_one_liner:
            print(
                f"{row['image_name']} -> set={row['prediction_set']} "
                f"(p_{class_labels[0]}={row[f'{class_labels[0]}_prob']:.3f}, "
                f"p_{class_labels[1]}={row[f'{class_labels[1]}_prob']:.3f}, "
                f"point={row['point_prediction']})"
            )

    # 6) CSV output
    if csv_path is not None and rows:
        os.makedirs(csv_path, exist_ok=True)
        csv_file = os.path.join(csv_path, csv_filename)
        file_exists = os.path.isfile(csv_file)
        fieldnames = list(rows[0].keys())
        with open(csv_file, mode="a" if file_exists else "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for r in rows:
                writer.writerow(r)

    return results
