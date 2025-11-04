
from __future__ import annotations
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin


@torch.no_grad()
def encode_and_normalize(model, imgs: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Encode a list of preprocessed images into CLIP embeddings and L2-normalize them.

    Args:
        model: CLIP-like model exposing encode_image.
        imgs: Sequence of image tensors produced by the CLIP preprocess pipeline.
              Each element should be shaped [1, 3, H, W] or [3, H, W].

    Returns:
        torch.Tensor: [N, D], L2-normalized.
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


def _is_feature_like(x: torch.Tensor, d: int) -> bool:
    """
    Decide if a tensor looks like a feature vector or a batch of features.

    Heuristics: 1D [D] or 2D [N, D] with trailing dim == D and without spatial dims.
    """
    if not torch.is_tensor(x):
        return False
    if x.ndim == 1:
        return x.shape[0] == d
    if x.ndim == 2:
        return x.shape[1] == d
    return False


def _stack_features(seq: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Stack a sequence of 1D feature tensors [D] into [N, D].
    If any element is 2D [1, D], squeeze the batch dim.
    """
    rows = []
    for t in seq:
        if t.ndim == 2 and t.shape[0] == 1:
            t = t.squeeze(0)
        rows.append(t)
    return torch.stack(rows, dim=0)


class CLIPWrapper(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible classifier for few-shot CLIP.

    Scores by cosine similarity to each class's few-shot bank:
    - For each class, take the max similarity to its bank as the class logit.
    - Apply temperature-scaled softmax to obtain probabilities.

    Now supports passing either images or precomputed features.
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
    def _logits_from_feature(self, tf: torch.Tensor) -> torch.Tensor:
        """Compute logits from a single normalized feature vector [D]."""
        # Ensure normalization just in case the caller supplied raw features
        tf = tf / tf.norm(dim=-1, keepdim=True)

        max_nom = torch.max(tf @ self.nominal_feats.T).item()
        if self.defective_feats is not None and self.defective_feats.nelement() > 0:
            max_def = torch.max(tf @ self.defective_feats.T).item()
        else:
            # If no defective bank is provided, make defective very unlikely.
            max_def = max_nom - 50.0

        sims = torch.tensor([max_nom, max_def], dtype=torch.float32)
        return sims / self.temperature

    @torch.no_grad()
    def _to_features(self, X: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """
        Convert inputs (images or features) to a feature matrix [N, D].

        Accepted forms:
          - torch.Tensor [N, D] or [D]
          - Sequence of feature tensors [D] or [1, D]
          - Sequence of image tensors [1,3,H,W] or [3,H,W] (will be encoded once)
        """
        D = int(self.nominal_feats.shape[1]) if self.nominal_feats.ndim == 2 else int(self.nominal_feats.shape[-1])

        if torch.is_tensor(X):
            if _is_feature_like(X, D):
                feats = X
                if feats.ndim == 1:
                    feats = feats.unsqueeze(0)
                # Re-normalize defensively
                return feats / feats.norm(dim=-1, keepdim=True)
            # Single image tensor
            return encode_and_normalize(self.model, [X])

        # Sequence path
        if len(X) == 0:
            return torch.empty(0, D)

        first = X[0]
        if torch.is_tensor(first) and _is_feature_like(first, D):
            feats = _stack_features(X)  # [N, D]
            return feats / feats.norm(dim=-1, keepdim=True)

        # Treat as images
        return encode_and_normalize(self.model, list(X))

    def predict_proba(self, X: Sequence[torch.Tensor] | torch.Tensor) -> np.ndarray:
        feats = self._to_features(X)  # [N, D]
        logits = torch.stack([self._logits_from_feature(tf) for tf in feats], dim=0)
        return F.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X: Sequence[torch.Tensor] | torch.Tensor) -> np.ndarray:
        proba = self.predict_proba(X)
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
