"""Tests for metrics module."""

import pytest
from conformal_clip import make_true_labels_from_counts


def test_make_true_labels_from_counts():
    """Test label expansion from counts."""
    labels = ["Nominal", "Defective"]
    label_counts = [3, 2]

    result = make_true_labels_from_counts(labels, label_counts)

    assert len(result) == 5
    assert result == ["Nominal", "Nominal", "Nominal", "Defective", "Defective"]


def test_make_true_labels_empty():
    """Test with empty inputs."""
    labels = []
    label_counts = []

    result = make_true_labels_from_counts(labels, label_counts)

    assert len(result) == 0
    assert result == []


def test_make_true_labels_single_class():
    """Test with single class."""
    labels = ["Nominal"]
    label_counts = [5]

    result = make_true_labels_from_counts(labels, label_counts)

    assert len(result) == 5
    assert all(label == "Nominal" for label in result)


def test_make_true_labels_multiple_classes():
    """Test with multiple classes."""
    labels = ["Class1", "Class2", "Class3"]
    label_counts = [2, 3, 1]

    result = make_true_labels_from_counts(labels, label_counts)

    assert len(result) == 6
    assert result[:2] == ["Class1", "Class1"]
    assert result[2:5] == ["Class2", "Class2", "Class2"]
    assert result[5:] == ["Class3"]
