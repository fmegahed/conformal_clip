"""
Shared Utilities for the Microstructure Examples
===============================================

This module provides shared data loading and splitting utilities for the
microstructure examples that use the qe_genai GitHub repository.

Functions:
    build_microstructure_splits: Create reproducible train/calibration/test
                                 splits for the microstructure dataset.
    load_pil_from_github: Load image URLs as PIL.Image objects.

Note: This module is only used by example scripts and is not part of the main
      conformal_clip package API.
"""

from io import BytesIO
from pathlib import Path
from typing import List, Tuple
import random

import requests
from PIL import Image

from conformal_clip.io_github import get_image_urls


def _list_imgs(p: str) -> List[str]:
    exts = {"jpg", "jpeg", "png"}
    return [str(q) for q in Path(p).iterdir() if q.suffix.lower().lstrip(".") in exts]


def build_microstructure_splits(
    seed: int = 2024,
    test_size: int = 9 * 25,   # matches pasted code
    learn_size: int = 8 * 9    # matches pasted code
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Build train / calibration / test splits for the microstructure dataset.

    Dataset structure
    -----------------
    One nominal class:
        - "Uniform"

    Nine defective subclasses:
        - "Band_high", "Band_low", "Band_medium"
        - "Bimodal_high", "Bimodal_low", "Bimodal_medium"
        - "SingleCrystal_high", "SingleCrystal_low", "SingleCrystal_medium"

    For benchmarking, we collapse all defective subclasses into a single
    "Defective" class, so this becomes a binary problem, similar to the
    extrusion example.

    Matching the original script
    ----------------------------
    With the default arguments, this reproduces our original selection logic:

        random.seed(2024)
        test_size = 9 * 25
        learn_size = 8 * 9

    - Test:
        * 225 Uniform images
        * 25 test images from each of the 9 defective subclasses
    - Learning pool (before splitting):
        * 72 Uniform "learning" images
        * 8 "learning" images from each defective subclass

    The only added step is that the learning pool for each class is split
    50/50 into train and calibration, with shuffling done *within each
    class only*. We do not globally shuffle the calibration set, so the
    per-class calibration distribution is preserved exactly.

    Returns
    -------
    train_nominal : list[str]
        Training image URLs for the nominal class.
    train_defective : list[str]
        Training image URLs for the defective (all non Uniform) class.
    calib_urls : list[str]
        Calibration image URLs (nominal + defective).
    calib_labels : list[str]
        Binary labels for calib_urls, "Nominal" or "Defective".
    test_urls : list[str]
        Test image URLs (nominal + defective).
    test_labels : list[str]
        Binary labels for test_urls, "Nominal" or "Defective".
    """
    repo_owner = "fmegahed"
    repo_name = "qe_genai"
    base_path = "data/microstructure"

    rng = random.Random(seed)

    # 1. Retrieve URLs for each subfolder (same structure as our original code)
    nominal_image_urls = get_image_urls(repo_owner, repo_name, base_path, "Uniform")

    defective_band_h_image_urls = get_image_urls(repo_owner, repo_name, base_path, "Band_high")
    defective_band_l_image_urls = get_image_urls(repo_owner, repo_name, base_path, "Band_low")
    defective_band_m_image_urls = get_image_urls(repo_owner, repo_name, base_path, "Band_medium")

    defective_bimodal_h_image_urls = get_image_urls(repo_owner, repo_name, base_path, "Bimodal_high")
    defective_bimodal_l_image_urls = get_image_urls(repo_owner, repo_name, base_path, "Bimodal_low")
    defective_bimodal_m_image_urls = get_image_urls(repo_owner, repo_name, base_path, "Bimodal_medium")

    defective_single_h_image_urls = get_image_urls(repo_owner, repo_name, base_path, "SingleCrystal_high")
    defective_single_l_image_urls = get_image_urls(repo_owner, repo_name, base_path, "SingleCrystal_low")
    defective_single_m_image_urls = get_image_urls(repo_owner, repo_name, base_path, "SingleCrystal_medium")

    def sample(urls: List[str], k: int) -> List[str]:
        # Use the same behavior as random.sample in our original script
        return rng.sample(urls, k)

    # ------------------------------------------------------------
    # Step 1: Test set (identical logic to our original script)
    # ------------------------------------------------------------
    test_nominal_size = test_size           # 9 * 25 = 225
    per_defective_test = int(test_size / 9) # 25 per defective subclass

    test_nominal_image_urls = sample(nominal_image_urls, test_nominal_size)

    test_defective_band_h_image_urls = sample(defective_band_h_image_urls, per_defective_test)
    test_defective_band_l_image_urls = sample(defective_band_l_image_urls, per_defective_test)
    test_defective_band_m_image_urls = sample(defective_band_m_image_urls, per_defective_test)

    test_defective_bimodal_h_image_urls = sample(defective_bimodal_h_image_urls, per_defective_test)
    test_defective_bimodal_l_image_urls = sample(defective_bimodal_l_image_urls, per_defective_test)
    test_defective_bimodal_m_image_urls = sample(defective_bimodal_m_image_urls, per_defective_test)

    test_defective_single_h_image_urls = sample(defective_single_h_image_urls, per_defective_test)
    test_defective_single_l_image_urls = sample(defective_single_l_image_urls, per_defective_test)
    test_defective_single_m_image_urls = sample(defective_single_m_image_urls, per_defective_test)

    test_defective_image_urls = (
        test_defective_band_h_image_urls +
        test_defective_band_l_image_urls +
        test_defective_band_m_image_urls +
        test_defective_bimodal_h_image_urls +
        test_defective_bimodal_l_image_urls +
        test_defective_bimodal_m_image_urls +
        test_defective_single_h_image_urls +
        test_defective_single_l_image_urls +
        test_defective_single_m_image_urls
    )

    test_urls = test_nominal_image_urls + test_defective_image_urls
    test_labels = (
        ["Nominal"] * len(test_nominal_image_urls) +
        ["Defective"] * len(test_defective_image_urls)
    )

    # ------------------------------------------------------------
    # Step 2: Remaining images after removing test set
    # ------------------------------------------------------------
    remaining_nominal_image_urls = [
        url for url in nominal_image_urls if url not in test_nominal_image_urls
    ]

    remaining_defective_band_h_image_urls = [
        url for url in defective_band_h_image_urls if url not in test_defective_band_h_image_urls
    ]
    remaining_defective_band_l_image_urls = [
        url for url in defective_band_l_image_urls if url not in test_defective_band_l_image_urls
    ]
    remaining_defective_band_m_image_urls = [
        url for url in defective_band_m_image_urls if url not in test_defective_band_m_image_urls
    ]

    remaining_defective_bimodal_h_image_urls = [
        url for url in defective_bimodal_h_image_urls if url not in test_defective_bimodal_h_image_urls
    ]
    remaining_defective_bimodal_l_image_urls = [
        url for url in defective_bimodal_l_image_urls if url not in test_defective_bimodal_l_image_urls
    ]
    remaining_defective_bimodal_m_image_urls = [
        url for url in defective_bimodal_m_image_urls if url not in test_defective_bimodal_m_image_urls
    ]

    remaining_defective_single_h_image_urls = [
        url for url in defective_single_h_image_urls if url not in test_defective_single_h_image_urls
    ]
    remaining_defective_single_l_image_urls = [
        url for url in defective_single_l_image_urls if url not in test_defective_single_l_image_urls
    ]
    remaining_defective_single_m_image_urls = [
        url for url in defective_single_m_image_urls if url not in test_defective_single_m_image_urls
    ]

    # ------------------------------------------------------------
    # Step 3: Shuffle remaining lists (per class, like our original script)
    # ------------------------------------------------------------
    rng.shuffle(remaining_nominal_image_urls)
    rng.shuffle(remaining_defective_band_h_image_urls)
    rng.shuffle(remaining_defective_band_l_image_urls)
    rng.shuffle(remaining_defective_band_m_image_urls)
    rng.shuffle(remaining_defective_bimodal_h_image_urls)
    rng.shuffle(remaining_defective_bimodal_l_image_urls)
    rng.shuffle(remaining_defective_bimodal_m_image_urls)
    rng.shuffle(remaining_defective_single_h_image_urls)
    rng.shuffle(remaining_defective_single_l_image_urls)
    rng.shuffle(remaining_defective_single_m_image_urls)

    # ------------------------------------------------------------
    # Step 4: Learning pool (identical to our original script)
    # ------------------------------------------------------------
    learn_nominal_image_urls = remaining_nominal_image_urls[:learn_size]  # 72

    per_defective_learn = int(learn_size / 9)  # 8 per defective subclass

    learn_defective_band_h_image_urls = remaining_defective_band_h_image_urls[:per_defective_learn]
    learn_defective_band_l_image_urls = remaining_defective_band_l_image_urls[:per_defective_learn]
    learn_defective_band_m_image_urls = remaining_defective_band_m_image_urls[:per_defective_learn]

    learn_defective_bimodal_h_image_urls = remaining_defective_bimodal_h_image_urls[:per_defective_learn]
    learn_defective_bimodal_l_image_urls = remaining_defective_bimodal_l_image_urls[:per_defective_learn]
    learn_defective_bimodal_m_image_urls = remaining_defective_bimodal_m_image_urls[:per_defective_learn]

    learn_defective_single_h_image_urls = remaining_defective_single_h_image_urls[:per_defective_learn]
    learn_defective_single_l_image_urls = remaining_defective_single_l_image_urls[:per_defective_learn]
    learn_defective_single_m_image_urls = remaining_defective_single_m_image_urls[:per_defective_learn]

    # ------------------------------------------------------------
    # Step 5: Split learning pool 50/50 into train and calib
    #         Shuffling is done PER CLASS. We never globally shuffle
    #         the combined calibration set, so class proportions are
    #         preserved exactly.
    # ------------------------------------------------------------
    train_nominal: List[str] = []
    train_defective: List[str] = []
    calib_urls: List[str] = []
    calib_labels: List[str] = []

    # Nominal: split within class
    rng.shuffle(learn_nominal_image_urls)
    half_nom = len(learn_nominal_image_urls) // 2
    train_nominal_urls = learn_nominal_image_urls[:half_nom]
    calib_nominal_urls = learn_nominal_image_urls[half_nom:]

    train_nominal.extend(train_nominal_urls)
    calib_urls.extend(calib_nominal_urls)
    calib_labels.extend(["Nominal"] * len(calib_nominal_urls))

    # Helper: split each defective subclass separately
    def split_defective_pool(pool: List[str]):
        rng.shuffle(pool)  # shuffle within this subclass only
        h = len(pool) // 2
        train_part = pool[:h]
        calib_part = pool[h:]
        train_defective.extend(train_part)
        calib_urls.extend(calib_part)
        calib_labels.extend(["Defective"] * len(calib_part))

    split_defective_pool(learn_defective_band_h_image_urls)
    split_defective_pool(learn_defective_band_l_image_urls)
    split_defective_pool(learn_defective_band_m_image_urls)

    split_defective_pool(learn_defective_bimodal_h_image_urls)
    split_defective_pool(learn_defective_bimodal_l_image_urls)
    split_defective_pool(learn_defective_bimodal_m_image_urls)

    split_defective_pool(learn_defective_single_h_image_urls)
    split_defective_pool(learn_defective_single_l_image_urls)
    split_defective_pool(learn_defective_single_m_image_urls)

    # Note: there is NO global shuffle of calib_urls / calib_labels here.
    # Class-wise proportions in the calibration set are preserved.

    return train_nominal, train_defective, calib_urls, calib_labels, test_urls, test_labels


def load_pil_from_github(urls: List[str]) -> List[Image.Image]:
    """Load a list of GitHub raw image URLs into PIL Image objects."""
    images: List[Image.Image] = []

    for url in urls:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        images.append(img)

    return images
