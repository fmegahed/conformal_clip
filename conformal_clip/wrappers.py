# conformal_clip/wrappers.py
from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin


@torch.no_grad()
def encode_and_normalize(model, imgs: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Encode a list of preprocessed images into CLIP embeddings and L2-normalize them.

    This wraps the model's encode_image and ensures each embedding has unit norm.
    It gracefully handles inputs shaped [1, 3, H, W] by removing the leading
    singleton batch dimension that CLIP often returns as [1, D].

    Args:
        model: CLIP-like model exposing encode_image.
        imgs: Sequence of image tensors produced by the CLIP preprocess pipeline.
              Each element should be shaped [1, 3, H, W] or [3, H, W].

    Returns:
        torch.Tensor: A 2D tensor of shape [N, D], where N is the number of images
        and D is the embedding dimension. Each row is L2-normalized.
    """
    outs = []
    for img in imgs:
        emb = model.encode_image(img)
        if emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb.squeeze(0)  # [D]
        outs.append(emb)
    feats = torch.stack(outs, dim=0)  # [N, D]
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


class CLIPWrapper(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible classifier for few-shot CLIP.

    It scores an input image by cosine similarity to each class's few-shot bank:
    - For each class, take the max similarity to its bank as the class logit.
    - Apply temperature-scaled softmax to obtain uncalibrated probabilities.

    The wrapper is intentionally lightweight and does not "train." It exists to
    provide a scikit-learn style interface: classes_, predict_proba, predict.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        model,
        nominal_feats: torch.Tensor,
        defective_feats: torch.Tensor | None,
        temperature: float = 1.0,
        class_labels: Sequence[str] = ("Nominal", "Defective"),
    ):
        self.model = model
        self.nominal_feats = nominal_feats
        self.defective_feats = defective_feats if defective_feats is not None else torch.empty(0)
        self.temperature = float(temperature)
        self.classes_ = np.array(list(class_labels))
        self._estimator_type = "classifier"  # also set at instance level

    def fit(self, X, y=None):
        # No fitting required. Provided for sklearn API compatibility.
        return self

    @torch.no_grad()
    def _logits_for_one(self, img) -> torch.Tensor:
        tf = self.model.encode_image(img)
        if tf.ndim == 2 and tf.shape[0] == 1:
            tf = tf.squeeze(0)
        tf = tf / tf.norm(dim=-1, keepdim=True)

        max_nom = torch.max(tf @ self.nominal_feats.T).item()
        if self.defective_feats is not None and self.defective_feats.nelement() > 0:
            max_def = torch.max(tf @ self.defective_feats.T).item()
        else:
            # If no defective bank is provided, make defective very unlikely.
            max_def = max_nom - 50.0

        sims = torch.tensor([max_nom, max_def], dtype=torch.float32)
        return sims / self.temperature

    def predict_proba(self, X_imgs: Sequence) -> np.ndarray:
        logits = torch.stack([self._logits_for_one(img) for img in X_imgs], dim=0)
        return F.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X_imgs: Sequence) -> np.ndarray:
        proba = self.predict_proba(X_imgs)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    # Minimal plumbing to keep sklearn.clone/get_params happy without copying tensors
    def get_params(self, deep: bool = True):
        return {
            "temperature": self.temperature,
            "class_labels": tuple(self.classes_.tolist()),
        }

    def set_params(self, **params):
        if "temperature" in params:
            self.temperature = float(params["temperature"])
        if "class_labels" in params:
            self.classes_ = np.array(list(params["class_labels"]))
        return self
