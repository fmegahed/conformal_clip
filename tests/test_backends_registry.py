from conformal_clip.backends import VISION_LANGUAGE_BACKENDS, VISION_ONLY_BACKENDS, load_backend
import pytest


def test_backend_keys_present():
    # Ensure expected keys are present to prevent accidental renames
    for k in [
        "openai",
        "openclipbase",
        "siglip2",
        "eva02",
        "mobileclip2",
        "vitg",
        "resnet50",
        "convnext",
        "coca",
        "custom-clip",
        "custom-clip-hf",
    ]:
        assert k in VISION_LANGUAGE_BACKENDS
    for k in ["dinov3", "mobilenetv4", "custom-vision"]:
        assert k in VISION_ONLY_BACKENDS


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        load_backend("does-not-exist", None, device="cpu")


def test_custom_without_id_raises():
    with pytest.raises(ValueError):
        load_backend("custom-clip", None, device="cpu")
    with pytest.raises(ValueError):
        load_backend("custom-vision", None, device="cpu")
