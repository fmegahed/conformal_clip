"""Tests for image_io module."""

import pytest
import numpy as np
from conformal_clip import load_image
from PIL import Image
import tempfile
import os


def test_load_image_invalid_input():
    """Test load_image with invalid input."""
    with pytest.raises(ValueError, match="must be a string"):
        load_image(123)  # Not a string


def test_load_image_nonexistent_file():
    """Test load_image with nonexistent file."""
    with pytest.raises(IOError):
        load_image("nonexistent_file_xyz123.jpg")


def test_load_image_from_file():
    """Test loading an image from a file."""
    # Create a temporary image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_path = f.name
        img = Image.new("RGB", (100, 100), color="red")
        img.save(temp_path)

    try:
        # Load the image
        loaded_img = load_image(temp_path)

        assert isinstance(loaded_img, Image.Image)
        assert loaded_img.mode == "RGB"
        assert loaded_img.size == (100, 100)
    finally:
        # Clean up
        os.unlink(temp_path)


def test_load_image_mode_conversion():
    """Test load_image with mode conversion."""
    # Create a temporary grayscale image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_path = f.name
        img = Image.new("L", (100, 100), color=128)
        img.save(temp_path)

    try:
        # Load as RGB
        loaded_img = load_image(temp_path, mode="RGB")

        assert isinstance(loaded_img, Image.Image)
        assert loaded_img.mode == "RGB"
        assert loaded_img.size == (100, 100)
    finally:
        os.unlink(temp_path)
