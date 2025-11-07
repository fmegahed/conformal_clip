"""Tests for wrappers module."""

import pytest
import torch
import numpy as np


def test_clip_wrapper_initialization():
    """Test CLIPWrapper can be initialized with dummy features."""
    from conformal_clip import CLIPWrapper

    # Create dummy features
    D = 512  # Feature dimension
    nominal_feats = torch.randn(5, D)
    nominal_feats = nominal_feats / nominal_feats.norm(dim=-1, keepdim=True)

    defective_feats = torch.randn(5, D)
    defective_feats = defective_feats / defective_feats.norm(dim=-1, keepdim=True)

    # Initialize wrapper (without actual CLIP model)
    wrapper = CLIPWrapper(
        model=None,  # Model not needed for this test
        nominal_feats=nominal_feats,
        defective_feats=defective_feats,
        temperature=1.0,
        class_labels=("Nominal", "Defective")
    )

    assert wrapper.temperature == 1.0
    assert len(wrapper.classes_) == 2
    assert wrapper.classes_[0] == "Nominal"
    assert wrapper.classes_[1] == "Defective"


def test_clip_wrapper_predict_with_features():
    """Test CLIPWrapper predictions using precomputed features."""
    from conformal_clip import CLIPWrapper

    # Create dummy features
    D = 512
    nominal_feats = torch.randn(5, D)
    nominal_feats = nominal_feats / nominal_feats.norm(dim=-1, keepdim=True)

    defective_feats = torch.randn(5, D)
    defective_feats = defective_feats / defective_feats.norm(dim=-1, keepdim=True)

    wrapper = CLIPWrapper(
        model=None,
        nominal_feats=nominal_feats,
        defective_feats=defective_feats,
        temperature=1.0,
        class_labels=("Nominal", "Defective")
    )

    # Create test features (similar to nominal)
    test_feat = nominal_feats[0].unsqueeze(0)

    # Test predict_proba
    proba = wrapper.predict_proba(test_feat)
    assert proba.shape == (1, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
    assert np.all(proba >= 0) and np.all(proba <= 1)  # Probabilities in [0, 1]

    # Test predict
    pred = wrapper.predict(test_feat)
    assert pred.shape == (1,)
    assert pred[0] in ["Nominal", "Defective"]


def test_clip_wrapper_sklearn_interface():
    """Test CLIPWrapper sklearn interface."""
    from conformal_clip import CLIPWrapper

    D = 512
    nominal_feats = torch.randn(5, D)
    nominal_feats = nominal_feats / nominal_feats.norm(dim=-1, keepdim=True)

    defective_feats = torch.randn(5, D)
    defective_feats = defective_feats / defective_feats.norm(dim=-1, keepdim=True)

    wrapper = CLIPWrapper(
        model=None,
        nominal_feats=nominal_feats,
        defective_feats=defective_feats,
        temperature=1.0,
        class_labels=("Nominal", "Defective")
    )

    # Test fit (should be no-op)
    result = wrapper.fit(None, None)
    assert result is wrapper

    # Test get_params
    params = wrapper.get_params()
    assert "temperature" in params
    assert params["temperature"] == 1.0

    # Test set_params
    wrapper.set_params(temperature=2.0)
    assert wrapper.temperature == 2.0
