"""
Shared Utilities for the Extrusion Examples
=========================================

This module provides shared data loading and splitting utilities for the extrusion
staple examples. 

Functions:
    build_extrusion_splits: Create reproducible train/calibration/test splits
    load_pil: Load image paths as PIL.Image objects

Note: This module is only used by example scripts and is not part of the main
      conformal_clip package API.
"""

import os
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests
from PIL import Image

from conformal_clip.io_github import get_image_urls




def _list_imgs(p: str) -> List[str]:
    exts = {"jpg", "jpeg", "png"}
    return [str(q) for q in Path(p).iterdir() if q.suffix.lower().lstrip(".") in exts]


def build_extrusion_splits(seed: int = 2025) -> Tuple[
    List[str], List[str], List[str], List[str], List[str], List[str]
]:
    """Return (train_nominal, train_defective, calib, calib_labels, test, test_labels) image paths.

    Keeps class balance simple: uses local+global as defective.
    """
    repo_owner = "fmegahed"
    repo_name = "qe_genai"
    base_path = "data/extrusion_images"
    
    rng = np.random.default_rng(seed)

    nom = get_image_urls(repo_owner, repo_name, base_path, "normal")
    nom.sort()
    over = get_image_urls(repo_owner, repo_name, base_path, "over")
    over.sort()
    under = get_image_urls(repo_owner, repo_name, base_path, "under")
    under.sort()

    def sample(paths, k):
        idx = rng.choice(len(paths), size=min(k, len(paths)), replace=False)
        return [paths[i] for i in idx]

    # Test set
    test_nom = sample(nom, 100)
    test_def = sample(over, 50) + sample(under, 50)
    test = test_nom + test_def
    test_labels = ["Nominal"] * len(test_nom) + ["Defective"] * len(test_def)

    # Remove used from pools
    nom_left = [p for p in nom if p not in test_nom]
    over_left = [p for p in over if p not in test_def]
    under_left = [p for p in under if p not in test_def]

    # Train exemplars (few-shot banks)
    tr_nom = sample(nom_left, 50)
    tr_def = sample(over_left, 25) + sample(under_left, 25)

    # Calibration set
    nom_left2 = [p for p in nom_left if p not in tr_nom]
    over_left2 = [p for p in over_left if p not in tr_def]
    under_left2 = [p for p in under_left if p not in tr_def]
    
    cal_nom = sample(nom_left2, 50)
    cal_def = sample(over_left2, 25) + sample(under_left2, 25)

    calib = cal_nom + cal_def
    calib_labels = ["Nominal"] * len(cal_nom) + ["Defective"] * len(cal_def)

    return tr_nom, tr_def, calib, calib_labels, test, test_labels


def load_pil_from_github(urls: List[str]) -> List[Image.Image]:
    images = []
    
    for url in urls:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        images.append(img)
    
    return images

