"""
Backends
========
This module centralizes all model loading for CLIP-like and vision-only backbones,
and exposes a simple API:

    model, preprocess_fn, tokenize_fn = load_backend(backend, backend_model_id, device)

where:
  - model has .encode_image(...), and for CLIP-like models .encode_text(...)
  - preprocess_fn takes a PIL.Image and returns a tensor on the correct device
  - tokenize_fn(label_list) returns token ids for CLIP-like models, or None
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple
import os

# Try to load .env so HF token can be supplied that way
try:  # pragma: no cover - optional convenience only
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    # If python-dotenv is not installed, we just skip
    pass

import torch

# In-process cache so repeated calls to load_backend with the same
# (backend, backend_model_id, device) reuse the same model object.
_BACKEND_CACHE: dict[tuple[str, str, str], tuple[torch.nn.Module, Callable, Optional[Callable]]] = {}

# Vision-language (CLIP-like) backends
VISION_LANGUAGE_BACKENDS = {
    # Smaller ViT CLIP variants
    "clip_b32": "ViT-B-32-quickgelu",
    "clip_b16": "ViT-B-16-quickgelu",
    # ~ Our previous implementation (OpenAI-trained CLIP)
    "openai": "ViT-L-14-quickgelu",
    # Likely OpenCLIP's best performing 224px encoder based on their README
    "openclipbase": "ViT-H-14-quickgelu",
    # ~ Best performing ViT
    "vitg": "ViT-bigG-14",
    # ResNet architecture within OpenCLIP, useful for benchmarking against the older convolutional paradigm
    "resnet50": "RN50x64-quickgelu",
    # EVA02 family model
    "eva02": "EVA02-E-14-plus",
    # ConvNeXt family encoder (very large)
    "convnext": "convnext_xxlarge",
    # Google's SIGLIP2 (Different Loss Function)
    "siglip2": "ViT-B-16-SigLIP2",
    # Apple MobileCLIP2 with OpenCLIP configs
    "mobileclip2": "MobileCLIP2-S4",
    # CoCa (Contrastive Captioners) models
    "coca": "coca_ViT-L-14",
    # Expert-specified another model from open_clip
    "custom-clip": None,
    # Expert-specified HF repo under open_clip ecosystem
    "custom-clip-hf": None,
}


# Vision only encoders (timm-based)
VISION_ONLY_BACKENDS = {
    "dinov3": "vit_large_patch16_dinov3.lvd1689m",
    "mobilenetv4": "mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k",
    # Additional small vision-only encoders
    "resnet18": "resnet18.a1_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    # Any timm-based vision-only model
    "custom-vision": None,
}


def _ensure_hf_auth_env():
    """Propagate HF token from HF_TOKEN to the variable used by huggingface_hub."""
    tok = os.getenv("HF_TOKEN")
    if tok and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = tok


def _pretrained_tags_by_model(open_clip) -> dict[str, set[str]]:
    """
    Build a mapping from model name -> set of pretrained tags in a
    version-agnostic way across open-clip-torch releases.

    Newer versions of open_clip may require arguments for certain helpers
    (e.g., list_pretrained_tags_by_model(model)), so we always fall back
    to list_pretrained() which is stable.
    """
    by_model: dict[str, set[str]] = {}

    # Preferred path for older APIs that still support the zero-arg helper
    fn = getattr(open_clip, "list_pretrained_tags_by_model", None)
    if callable(fn):
        try:
            res = fn()  # type: ignore[call-arg]
        except TypeError:
            # Signature has changed; ignore and use list_pretrained() instead.
            res = None
        else:
            if isinstance(res, dict):
                return {k: set(v) for k, v in res.items()}

    # Fallback: use list_pretrained() which returns (model, tag, *rest)
    all_tags = open_clip.list_pretrained()
    for item in all_tags:
        if isinstance(item, tuple) and len(item) >= 2:
            name, tag = item[0], item[1]
            by_model.setdefault(name, set()).add(tag)
    return by_model


def _choose_pretrained_tag(open_clip, model_name: str) -> str:
    """Pick a reasonable pretrained tag for a given built-in model name."""
    by_model = _pretrained_tags_by_model(open_clip)
    tags = list(by_model.get(model_name, []))
    if not tags:
        # Try global list_pretrained for safety
        all_tags = open_clip.list_pretrained()
        # Fall back to any tag that mentions the model
        for tag in all_tags:
            if isinstance(tag, tuple) and tag[0] == model_name:
                tags.append(tag[1])
    if not tags:
        # Last resort
        return "openai"
    # Prefer openai, then laion2b, else first
    for pref in ("openai", "laion2b"):
        for t in tags:
            if pref in t:
                return t
    return tags[0]


def _load_openclip_builtin(model_name: str, device: torch.device | str, pretrained_tag: str | None = None):
    import open_clip

    tag = pretrained_tag or _choose_pretrained_tag(open_clip, model_name)
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name, pretrained=tag
    )
    preprocess = preprocess_val

    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    model.eval()

    get_tok = getattr(open_clip, "get_tokenizer", None)
    if callable(get_tok):
        tokenizer = get_tok(model_name)

        def tokenize_fn(text_list):
            return tokenizer(text_list).to(device)
    else:
        from open_clip import tokenize as oc_tokenize  # type: ignore

        def tokenize_fn(text_list):
            return oc_tokenize(text_list).to(device)

    def preprocess_fn(pil_image):
        t = preprocess(pil_image)
        return t.to(device)

    return model, preprocess_fn, tokenize_fn


def _load_openclip_hf(hf_id: str, device: torch.device | str):
    """Load a CLIP-like model from HF via OpenCLIP using an hf-hub identifier."""
    _ensure_hf_auth_env()
    import open_clip

    obj = open_clip.create_model_from_pretrained(hf_id)
    if isinstance(obj, tuple):
        if len(obj) == 2:
            model, preprocess = obj
        else:
            model, preprocess_train, preprocess_val = obj
            preprocess = preprocess_val
    else:
        # If transforms are not returned, create default transforms
        model = obj
        _, preprocess_train, preprocess = open_clip.create_model_and_transforms(model.visual.__class__.__name__)

    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    model.eval()

    get_tok = getattr(open_clip, "get_tokenizer", None)
    if callable(get_tok):
        # Try to infer tokenizer from hf_id, else fallback to generic clip tokenizer
        try:
            tokenizer = get_tok(hf_id)
        except Exception:
            tokenizer = get_tok("ViT-L-14-quickgelu")

        def tokenize_fn(text_list):
            return tokenizer(text_list).to(device)
    else:
        from open_clip import tokenize as oc_tokenize  # type: ignore

        def tokenize_fn(text_list):
            return oc_tokenize(text_list).to(device)

    def preprocess_fn(pil_image):
        t = preprocess(pil_image)
        return t.to(device)

    return model, preprocess_fn, tokenize_fn


class _VisionOnlyAdapter(torch.nn.Module):
    """Adapter that provides encode_image(...) over a timm vision backbone.

    The underlying timm model is created with num_classes=0 to emit pooled features.
    """

    def __init__(self, timm_model: torch.nn.Module):
        super().__init__()
        self.backbone = timm_model

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure batch input
        if x.ndim == 3:
            x = x.unsqueeze(0)
        feats = self.backbone(x)
        return feats


def _load_timm(model_name: str, device: torch.device | str):
    import timm
    from timm.data import resolve_data_config, create_transform

    _ensure_hf_auth_env()

    m = timm.create_model(model_name, pretrained=True, num_classes=0)
    m.eval()

    if isinstance(device, str):
        device = torch.device(device)
    m = m.to(device)

    cfg = resolve_data_config({}, model=m)
    transform = create_transform(**cfg, is_training=False)

    adapter = _VisionOnlyAdapter(m)

    def preprocess_fn(pil_image):
        t = transform(pil_image)
        return t.to(device)

    tokenize_fn = None
    return adapter, preprocess_fn, tokenize_fn


def load_backend(
    backend: str,
    backend_model_id: Optional[str],
    device: torch.device | str = "cpu",
) -> Tuple[torch.nn.Module, Callable, Optional[Callable]]:
    """Load a model backend along with preprocess and tokenizer functions.

    Args:
        backend: One of VISION_LANGUAGE_BACKENDS keys or VISION_ONLY_BACKENDS keys.
        backend_model_id: Optional override / custom id. For "custom-clip" and
            "custom-vision" this must be provided.
        device: Torch device or string, e.g., "cuda", torch.device("cuda:0").

    Returns:
        (model, preprocess_fn, tokenize_fn)
    """
    backend = (backend or "").lower()
    dev_str = str(device)
    cache_key = (backend, backend_model_id or "", dev_str)

    cached = _BACKEND_CACHE.get(cache_key)
    if cached is not None:
        return cached

    model: torch.nn.Module
    preprocess_fn: Callable
    tokenize_fn: Optional[Callable]

    if backend in VISION_LANGUAGE_BACKENDS:
        default_val = VISION_LANGUAGE_BACKENDS[backend]
        if backend == "custom-clip-hf":
            if not backend_model_id:
                raise ValueError("For backend 'custom-clip-hf', provide an hf-hub id, e.g., 'hf-hub:org/repo'.")
            model, preprocess_fn, tokenize_fn = _load_openclip_hf(backend_model_id, device)

        elif backend == "custom-clip":
            if not backend_model_id:
                raise ValueError("For backend 'custom-clip', provide a model name or 'model@tag'.")
            # Support 'model@tag' format
            if "@" in backend_model_id:
                model_name, tag = backend_model_id.split("@", 1)
            else:
                model_name, tag = backend_model_id, None
            model, preprocess_fn, tokenize_fn = _load_openclip_builtin(model_name, device, pretrained_tag=tag)

        else:
            # Known built-ins: use model name from mapping; prefer a sensible pretrained tag
            model_name = default_val
            model, preprocess_fn, tokenize_fn = _load_openclip_builtin(model_name, device, pretrained_tag=None)

    elif backend in VISION_ONLY_BACKENDS:
        model_name = backend_model_id or VISION_ONLY_BACKENDS[backend]
        if model_name is None:
            raise ValueError(
                "For backend 'custom-vision', you must provide backend_model_id (timm model name)."
            )
        model, preprocess_fn, tokenize_fn = _load_timm(model_name, device)

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Known CLIP-like: {list(VISION_LANGUAGE_BACKENDS.keys())}; "
            f"vision-only: {list(VISION_ONLY_BACKENDS.keys())}"
        )

    _BACKEND_CACHE[cache_key] = (model, preprocess_fn, tokenize_fn)
    return model, preprocess_fn, tokenize_fn
