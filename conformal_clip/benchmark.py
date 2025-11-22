"""
Benchmarking
============
Systematic benchmarking across multiple backends, calibration methods, and conformal settings.

This module provides a unified interface to compare different model backends
(OpenCLIP, timm) with various calibration methods and conformal prediction modes
on the same train/calibration/test splits.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from .backends import load_backend, VISION_LANGUAGE_BACKENDS, VISION_ONLY_BACKENDS
from .wrappers import CLIPWrapper, encode_and_normalize
from .conformal import few_shot_fault_classification_conformal


def _ensure_2_classes(class_labels: Sequence[str]):
    uniq = list(dict.fromkeys(class_labels))
    if len(uniq) != 2:
        raise ValueError("This benchmarking utility currently supports exactly 2 classes.")
    return tuple(uniq)


def _prep_images(preprocess_fn: Callable, imgs: Sequence, device: torch.device | str) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for im in imgs:
        if torch.is_tensor(im):
            t = im.to(device)
        else:
            t = preprocess_fn(im)  # already moved to device by preprocess_fn
        out.append(t)
    return out


def _calibrate_estimator(
    estimator: CLIPWrapper,
    calib_feats: torch.Tensor,
    calib_labels: Sequence[str],
    class_labels: Tuple[str, str],
    method: Optional[str],
) -> CLIPWrapper:
    if method not in {None, "isotonic", "sigmoid"}:
        raise ValueError("Calibration method must be one of: None, 'isotonic', 'sigmoid'")

    if method is None:
        return estimator

    p_cal_uncal = estimator.predict_proba(calib_feats)[:, 1]
    y_cal = np.array([1 if lab == class_labels[1] else 0 for lab in calib_labels], dtype=int)

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal_uncal, y_cal)

        def transform_fn(p):
            p = np.asarray(p)
            return iso.predict(p)

    else:  # "sigmoid"
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

    class _Calibrated(CLIPWrapper):  # type: ignore[misc]
        def __init__(self, base: CLIPWrapper, tfm):
            self.base = base
            self.transform_fn = tfm
            self.classes_ = base.classes_
            self._estimator_type = "classifier"

        def predict_proba(self, X):
            p = self.base.predict_proba(X)
            p_pos = self.transform_fn(p[:, 1])
            p_pos = np.clip(p_pos, 1e-8, 1 - 1e-8)
            return np.column_stack([1.0 - p_pos, p_pos])

        def predict(self, X):
            p = self.predict_proba(X)
            return self.classes_[np.argmax(p, axis=1)]

    return _Calibrated(estimator, transform_fn)


def _point_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Tuple[str, str],
    prob_pos: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    pos = labels[1]
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, pos_label=pos, average="binary", zero_division=0)
    prec = precision_score(y_true, y_pred, pos_label=pos, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos, average="binary", zero_division=0)

    # Specificity from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc = float("nan")
    if prob_pos is not None:
        y_true_bin = [1 if t == pos else 0 for t in y_true]
        try:
            auc = roc_auc_score(y_true_bin, list(prob_pos))
        except Exception:
            auc = float("nan")

    return {
        "Accuracy": acc,
        "Sensitivity (Recall)": rec,
        "Specificity": spec,
        "Precision": prec,
        "F1 Score": f1,
        "AUC": auc,
    }


def _parse_set(s: str) -> List[str]:
    if isinstance(s, str) and s.strip().upper() == "ABSTAIN":
        return []
    if isinstance(s, str) and len(s.strip()) > 0:
        return [t.strip() for t in s.split("|")]
    return []


def _conformal_metrics(
    y_true: Sequence[str],
    pred_sets: Sequence[Sequence[str]],
    labels: Tuple[str, str],
) -> Dict[str, float]:
    contains_true = [t in s for t, s in zip(y_true, pred_sets)]
    coverage = float(np.mean(contains_true)) if contains_true else float("nan")
    per_class = {}
    for c in labels:
        idx = [i for i, t in enumerate(y_true) if t == c]
        per_class[c] = float(np.mean([contains_true[i] for i in idx])) if idx else float("nan")
    return {
        "Coverage (overall)": coverage,
        f"Coverage [{labels[0]}]": per_class[labels[0]],
        f"Coverage [{labels[1]}]": per_class[labels[1]],
    }


def _highlight_max_yellow(df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.io.formats.style.Styler:
    return df.style.highlight_max(subset=metric_cols, color="#FFF59D")


def benchmark_models(
    train_nominal_images: Sequence,
    train_defective_images: Sequence,
    calib_images: Sequence,
    calib_labels: Sequence[str],
    test_images: Sequence,
    test_labels: Sequence[str],
    device: torch.device | str = "cpu",
    temperature: float = 1.0,
    seed: int = 2025,
    backends: Optional[List[str]] = None,
    resource_tier: Optional[str] = "low",
    custom_backend_ids: Optional[Dict[str, str]] = None,
    calibration_methods: Sequence[Optional[str]] = (None, "isotonic", "sigmoid"),
    conformal_modes: Sequence[Optional[str]] = (None, "global", "mondrian"),
    alpha_list: Sequence[float] = (0.1,),
    allow_empty: bool = False,
    csv_path: Optional[str] = None,
    csv_prefix: Optional[str] = "",
    csv_filename_template: str = "{backend}_{cal}_{conf}.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.io.formats.style.Styler, pd.io.formats.style.Styler]:
    """
    Compare available backends across calibration and conformal configurations on fixed splits.

    Returns two DataFrames (classification metrics and conformal metrics) and their styled variants
    where the top value in each metric column is highlighted in yellow.

    CSV Parameters:
    ---------------
    csv_path : Optional[str], default None
        Directory where conformal prediction results are written as CSV files.
        If None, no CSV files are saved.
        When enabled, the function may generate multiple CSV files because each
        combination of backend, calibration method, conformal mode, and alpha
        produces a separate output file.

    csv_prefix : str, default ""
        Optional prefix added to every generated CSV filename.
        This allows users to group or tag runs without changing the underlying
        naming pattern.

    csv_filename_template : str, default "{backend}_{cal}_{conf}.csv"
        Template used to construct core filenames for the CSV outputs.
        The structure of the filename remains stable because the template
        defines the required pattern. The following template fields are available:
            * backend : backend identifier
            * cal : calibration method ("none", "isotonic", "sigmoid", etc.)
            * conf : conformal mode ("global" or "mondrian")
        The prefix provided through csv_prefix is prepended to this template.

    Notes:
      - Supports exactly two classes.
      - Vision-only backends participate in few-shot classification; zero-shot is not included here.
      - If backends is None, resource_tier can be used to restrict which models are benchmarked:
          * \"low\": only smaller / lightweight models suitable for low-resource environments
          * \"medium\": low-tier models plus mid-sized models that fit on ~8–12 GB GPUs
          * \"high\": all non-custom backends (default when resource_tier is None)
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensuring folder exist if not none
    if csv_path is not None:
        os.makedirs(csv_path, exist_ok=True)

    # Resolve class labels
    uniq = list(dict.fromkeys(list(calib_labels) + list(test_labels)))
    class_labels = _ensure_2_classes(uniq)

    # Backends to try (exclude custom keys unless explicitly provided)
    vl_keys = [k for k in VISION_LANGUAGE_BACKENDS.keys() if k != "custom-clip"]
    vo_keys = [k for k in VISION_ONLY_BACKENDS.keys() if k != "custom-vision"]
    default_backends = vl_keys + vo_keys
    if backends is not None:
        to_try = backends
    else:
        tier = (resource_tier or "").lower()
        low_set = {
            # CLIP-like small models
            "clip_b32",
            "clip_b16",
            "siglip2",
            "mobileclip2",
            # Vision-only small models
            "mobilenetv4",
            "resnet18",
            "efficientnet_b0",
        }
        medium_extra = {
            "openai",
            "resnet50",
            "coca",
            "dinov3",
        }
        high_extra = {
            "openclipbase",
            "vitg",
            "eva02",
            "convnext",
        }
        if tier == "low":
            allowed = low_set
        elif tier == "medium":
            allowed = low_set | medium_extra
        elif tier == "high":
            allowed = low_set | medium_extra | high_extra
        else:
            raise ValueError("resource_tier must be one of: 'low', 'medium', 'high'")
        # Preserve default ordering but restrict to allowed set
        to_try = [b for b in default_backends if b in allowed]

    # Storage
    cls_rows: List[Dict] = []
    cp_rows: List[Dict] = []

    # Iterate backends
    for be in to_try:
        be_id = None
        if custom_backend_ids and be in custom_backend_ids:
            be_id = custom_backend_ids[be]

        try:
            model, preprocess_fn, _tok = load_backend(be, be_id, device)
        except Exception as e:
            # Skip backend if it fails to load
            cls_rows.append({
                "backend": be,
                "calibration": None,
                "conformal": None,
                "alpha": float("nan"),
                "Error": f"{type(e).__name__}: {e}",
            })
            continue

        # Preprocess all splits (kept on device)
        nom_t = _prep_images(preprocess_fn, train_nominal_images, device)
        def_t = _prep_images(preprocess_fn, train_defective_images, device)
        cal_t = _prep_images(preprocess_fn, calib_images, device)
        tst_t = _prep_images(preprocess_fn, test_images, device)

        # Pre-encode banks and splits once
        nominal_feats = encode_and_normalize(model, nom_t)
        defective_feats = encode_and_normalize(model, def_t) if len(def_t) > 0 else torch.empty(0)
        calib_feats = encode_and_normalize(model, cal_t)
        test_feats = encode_and_normalize(model, tst_t)

        base_est = CLIPWrapper(
            model=model,
            nominal_feats=nominal_feats,
            defective_feats=defective_feats,
            temperature=temperature,
            class_labels=class_labels,
        )

        y_true = list(test_labels)

        for cal_m in calibration_methods:
            est = _calibrate_estimator(base_est, calib_feats, calib_labels, class_labels, cal_m)

            # No conformal: point metrics only
            if None in conformal_modes:
                proba = est.predict_proba(test_feats)
                y_pred = est.predict(test_feats).tolist()
                prob_pos = proba[:, 1].tolist()
                m = _point_metrics(y_true, y_pred, class_labels, prob_pos)
                row = {"backend": be, "calibration": cal_m or "none", "conformal": "none", "alpha": float("nan")}
                row.update(m)
                cls_rows.append(row)

            # Conformal variants
            for conf in conformal_modes:
                # build the file name only if saving is enabled
                if csv_path is not None:
                    filename_core = csv_filename_template.format(
                        backend=be,
                        cal=cal_m or "none",
                        conf=conf,
                    )
                    csv_filename = f"{csv_prefix}{filename_core}"
                else:
                    csv_filename = None
                if conf is None:
                    continue
                for alpha in alpha_list:
                    # Use library function for consistency
                    results = few_shot_fault_classification_conformal(
                        model=model,
                        test_images=tst_t,
                        test_image_filenames=[f"img_{i}" for i in range(len(tst_t))],
                        nominal_images=nom_t,
                        nominal_descriptions=[f"Nominal: nominal_{i}" for i in range(max(1, len(nom_t)))],
                        defective_images=def_t,
                        defective_descriptions=[f"Defective: defective_{i}" for i in range(max(1, len(def_t)))],
                        calib_images=cal_t,
                        calib_labels=calib_labels,
                        alpha=float(alpha),
                        temperature=temperature,
                        mondrian=(conf == "mondrian"),
                        class_labels=class_labels,
                        print_one_liner=False,
                        seed=seed,
                        prob_calibration=(cal_m if cal_m is not None else None),
                        allow_empty=allow_empty,
                        csv_path=csv_path,
                        csv_filename=csv_filename,
                    )

                    # Classification metrics from results
                    y_pred = [r["point_prediction"] for r in results]
                    prob_pos = [r[f"{class_labels[1]}_prob"] for r in results]
                    m = _point_metrics(y_true, y_pred, class_labels, prob_pos)
                    row = {
                        "backend": be,
                        "calibration": cal_m or "none",
                        "conformal": conf,
                        "alpha": float(alpha),
                    }
                    row.update(m)
                    cls_rows.append(row)

                    # Conformal metrics
                    pred_sets = [_parse_set(r["prediction_set"]) for r in results]
                    cm = _conformal_metrics(y_true, pred_sets, class_labels)
                    row2 = {
                        "backend": be,
                        "calibration": cal_m or "none",
                        "conformal": conf,
                        "alpha": float(alpha),
                    }
                    row2.update(cm)
                    cp_rows.append(row2)

    cls_df = pd.DataFrame(cls_rows)
    cp_df = pd.DataFrame(cp_rows)

    # Highlight best per metric in yellow
    cls_metric_cols = [c for c in ["Accuracy", "Sensitivity (Recall)", "Specificity", "Precision", "F1 Score", "AUC"] if c in cls_df.columns]
    cp_metric_cols = [c for c in ["Coverage (overall)", f"Coverage [{class_labels[0]}]", f"Coverage [{class_labels[1]}]"] if c in cp_df.columns]

    cls_style = _highlight_max_yellow(cls_df, cls_metric_cols) if len(cls_metric_cols) else cls_df.style
    cp_style = _highlight_max_yellow(cp_df, cp_metric_cols) if len(cp_metric_cols) else cp_df.style

    return cls_df, cp_df, cls_style, cp_style
