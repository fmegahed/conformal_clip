
from __future__ import annotations
from typing import List, Dict, Any, Sequence
import os
import csv
from datetime import datetime

import numpy as np
import torch
from PIL import Image

from .wrappers import CLIPWrapper, encode_and_normalize


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Conservative finite-sample quantile used by conformal prediction.

    Uses ceil((1 - alpha) * (n + 1)) with higher-style selection.

    Args:
        scores: Array of nonconformity scores.
        alpha: Miscoverage rate in (0, 1).

    Returns:
        Quantile threshold as a float.
    """
    n = len(scores)
    if n <= 0:
        raise ValueError("scores must be non-empty")
    k = int(np.ceil((1.0 - alpha) * (n + 1)))
    k = min(max(k, 1), n)
    return float(np.partition(scores, k - 1)[k - 1])


def _fit_global_threshold(
    estimator,
    calib_images: Sequence[Image.Image],
    calib_labels: Sequence[str],
    alpha: float
) -> float:
    """
    Compute a single global threshold q on scores s = 1 - p_true across all classes.

    Args:
        estimator: Any classifier with predict_proba and classes_.
        calib_images: Calibration images.
        calib_labels: True labels.
        alpha: Miscoverage rate.

    Returns:
        Scalar threshold q.
    """
    probs = estimator.predict_proba(list(calib_images))  # shape [n_cal, C]
    y = np.array(list(calib_labels))
    idx_true = np.array([np.where(estimator.classes_ == lab)[0][0] for lab in y])
    p_true = probs[np.arange(len(y)), idx_true]
    s = 1.0 - p_true
    return _finite_sample_quantile(s, alpha)


def _fit_mondrian_thresholds(
    estimator,
    calib_images: Sequence[Image.Image],
    calib_labels: Sequence[str],
    alpha: float
) -> Dict[str, float]:
    """
    Compute class-conditional thresholds q_y on scores s = 1 - p_true.

    Args:
        estimator: Any classifier with predict_proba and classes_.
        calib_images: Calibration images.
        calib_labels: Class labels for each calibration image.
        alpha: Miscoverage rate.

    Returns:
        Mapping from class label -> threshold q_y.
    """
    probs = estimator.predict_proba(list(calib_images))  # [n_cal, C]
    y = np.array(list(calib_labels))
    idx_true = np.array([np.where(estimator.classes_ == lab)[0][0] for lab in y])
    p_true = probs[np.arange(len(y)), idx_true]
    s = 1.0 - p_true

    q_map: Dict[str, float] = {}
    for cls in estimator.classes_:
        mask = (y == cls)
        if not np.any(mask):
            continue
        q_map[str(cls)] = _finite_sample_quantile(s[mask], alpha)

    # Fallback for classes unseen in calibration
    if len(q_map) < len(estimator.classes_):
        q_global = _finite_sample_quantile(s, alpha)
        for cls in estimator.classes_:
            q_map.setdefault(str(cls), q_global)

    return q_map


def _predict_sets_global(
    estimator,
    X_imgs,
    q: float,
    allow_empty: bool = False
):
    probs = estimator.predict_proba(list(X_imgs))  # [n, C]
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
    X_imgs,
    q_map: dict[str, float],
    allow_empty: bool = False
):
    probs = estimator.predict_proba(list(X_imgs))  # [n, C]
    sets = []
    for i in range(probs.shape[0]):
        inc = []
        for j, cls in enumerate(estimator.classes_):
            q = q_map[str(cls)]
            if probs[i, j] >= 1.0 - q:
                inc.append(str(cls))
        if not allow_empty and len(inc) == 0:
            j_star = int(np.argmax(probs[i]))
            inc = [str(estimator.classes_[j_star])]
        sets.append(inc)
    return sets


def _encode_one_image_feature(model, img: Image.Image) -> torch.Tensor:
    """
    Encode a single image and return a 1D normalized feature tensor [D].

    Handles batched outputs like [1, D] by squeezing.
    """
    with torch.no_grad():
        tf = model.encode_image(img)
        if tf.ndim == 2 and tf.shape[0] == 1:
            tf = tf.squeeze(0)
        tf = tf / tf.norm(dim=-1, keepdim=True)
    return tf


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

    Probability calibration options:
      - None: use softmax over cosine-similarity logits (status quo)
      - "isotonic": nonparametric monotone calibration via sklearn.isotonic.IsotonicRegression
      - "sigmoid":  Platt-like calibration via sklearn.linear_model.LogisticRegression on logit(p_uncal)

    Conformal thresholds are computed on the (possibly calibrated) probabilities.
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

    # ---- Probability calibration (no CalibratedClassifierCV) -----------------
    class _CalibratedProbaWrapper:
        def __init__(self, base_est, transform_fn, classes_):
            self.base_est = base_est
            self.transform_fn = transform_fn  # maps p_uncal_pos -> p_cal_pos
            self.classes_ = np.array(classes_)
            self._estimator_type = "classifier"

        def predict_proba(self, X):
            # X must be a list/sequence of torch tensors (no numpy wrapping)
            p = self.base_est.predict_proba(list(X))      # [n, 2]
            p_pos_uncal = p[:, 1]
            p_pos = self.transform_fn(p_pos_uncal)
            p_pos = np.clip(p_pos, 1e-8, 1 - 1e-8)
            return np.column_stack([1.0 - p_pos, p_pos])

        def predict(self, X):
            p = self.predict_proba(list(X))
            idx = np.argmax(p, axis=1)
            return self.classes_[idx]

    estimator = clip_est
    if prob_calibration in {"isotonic", "sigmoid"}:
        # Uncalibrated positive-class probabilities on calibration split
        p_cal_uncal = clip_est.predict_proba(list(calib_images))[:, 1]
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

    # 3) Conformal thresholds on calibrated probabilities
    if mondrian:
        q_map = _fit_mondrian_thresholds(estimator, list(calib_images), list(calib_labels), alpha)
        pred_sets = _predict_sets_mondrian(estimator, list(test_images), q_map, allow_empty=allow_empty)
    else:
        q_global = _fit_global_threshold(estimator, list(calib_images), list(calib_labels), alpha)
        pred_sets = _predict_sets_global(estimator, list(test_images), q_global, allow_empty=allow_empty)

    # 4) Point predictions and probabilities from same estimator
    y_point = estimator.predict(list(test_images))
    proba   = estimator.predict_proba(list(test_images))  # [n, 2]

    # 5) Per-image traceability and CSV accumulation
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for idx, img in enumerate(test_images):
        tf = _encode_one_image_feature(model, img)
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
    if csv_path is not None:
        os.makedirs(csv_path, exist_ok=True)
        csv_file = os.path.join(csv_path, csv_filename)
        file_exists = os.path.isfile(csv_file)
        fieldnames = list(rows[0].keys()) if rows else []
        with open(csv_file, mode="a" if file_exists else "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists and fieldnames:
                writer.writeheader()
            for r in rows:
                writer.writerow(r)

    return results
