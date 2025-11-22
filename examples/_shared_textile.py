"""
Shared Utilities for Textile Examples
======================================

This module provides shared data loading and splitting utilities for the textile
defect classification examples. It requires the optional `conformal-clip[data]`
package which provides the example textile dataset.

Functions:
    build_textile_splits: Create reproducible train/calibration/test splits
    load_pil: Load image paths as PIL.Image objects

Note: This module is only used by example scripts and is not part of the main
      conformal_clip package API.
"""

import os
from pathlib import Path
from typing import List, Tuple
import numpy as np

from PIL import Image

try:
    from conformal_clip_data import nominal_dir, local_dir, global_dir
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "conformal_clip_data is required for these examples. Install with: pip install 'conformal-clip[data]'"
    ) from e


def _list_imgs(p: str) -> List[str]:
    exts = {"jpg", "jpeg", "png"}
    return [str(q) for q in Path(p).iterdir() if q.suffix.lower().lstrip(".") in exts]


def build_textile_splits(seed: int = 2025) -> Tuple[
    List[str], List[str], List[str], List[str], List[str], List[str]
]:
    """Return (train_nominal, train_defective, calib, calib_labels, test, test_labels) image paths.

    Keeps class balance simple: uses local+global as defective.
    """
    rng = np.random.default_rng(seed)

    nom = _list_imgs(nominal_dir())
    loc = _list_imgs(local_dir())
    glo = _list_imgs(global_dir())

    def sample(paths, k):
        idx = rng.choice(len(paths), size=min(k, len(paths)), replace=False)
        return [paths[i] for i in idx]

    # Test set
    test_nom = sample(nom, 50)
    test_def = sample(loc, 25) + sample(glo, 25)
    test = test_nom + test_def
    test_labels = ["Nominal"] * len(test_nom) + ["Defective"] * len(test_def)

    # Remove used from pools
    nom_left = [p for p in nom if p not in test_nom]
    loc_left = [p for p in loc if p not in test_def]
    glo_left = [p for p in glo if p not in test_def]

    # Train exemplars (few-shot banks)
    tr_nom = sample(nom_left, 50)
    tr_def = sample(loc_left, 25) + sample(glo_left, 25)

    # Calibration set
    nom_left2 = [p for p in nom_left if p not in tr_nom]
    loc_left2 = [p for p in loc_left if p not in tr_def]
    glo_left2 = [p for p in glo_left if p not in tr_def]
    cal_nom = sample(nom_left2, 50)
    cal_def = sample(loc_left2, 25) + sample(glo_left2, 25)
    calib = cal_nom + cal_def
    calib_labels = ["Nominal"] * len(cal_nom) + ["Defective"] * len(cal_def)

    return tr_nom, tr_def, calib, calib_labels, test, test_labels


def load_pil(paths: List[str]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]

