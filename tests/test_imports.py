"""Test that all public API imports work correctly."""

import pytest


def test_import_main_package():
    """Test importing the main package."""
    import conformal_clip
    assert conformal_clip.__version__ is not None


def test_import_all_public_functions():
    """Test importing all functions from __all__."""
    from conformal_clip import (
        get_image_urls,
        load_image,
        evaluate_zero_shot_predictions,
        CLIPWrapper,
        encode_and_normalize,
        few_shot_fault_classification_conformal,
        compute_classification_metrics,
        compute_conformal_set_metrics,
        make_true_labels_from_counts,
        plot_confusion_matrix,
    )

    # Verify functions are callable
    assert callable(get_image_urls)
    assert callable(load_image)
    assert callable(evaluate_zero_shot_predictions)
    assert callable(encode_and_normalize)
    assert callable(few_shot_fault_classification_conformal)
    assert callable(compute_classification_metrics)
    assert callable(compute_conformal_set_metrics)
    assert callable(make_true_labels_from_counts)
    assert callable(plot_confusion_matrix)


def test_clip_wrapper_class():
    """Test that CLIPWrapper can be imported and is a class."""
    from conformal_clip import CLIPWrapper
    assert isinstance(CLIPWrapper, type)


def test_version_format():
    """Test that version string is properly formatted."""
    import conformal_clip
    version = conformal_clip.__version__
    assert isinstance(version, str)
    parts = version.split(".")
    assert len(parts) >= 2  # At least major.minor
