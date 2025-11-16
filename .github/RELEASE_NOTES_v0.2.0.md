# Release v0.2.0 - Backends, Benchmarking, and Resource Tiers

## Overview

This release introduces a centralized backend loader for CLIP-like and timm models, a flexible benchmarking utility, and resource-aware defaults so users on low-resource machines are not forced to load very large models.

## Highlights

### Backend Loader

- Added `conformal_clip.backends` with:
  - `load_backend(backend, backend_model_id, device)` to load CLIP-like models via `open-clip-torch` and vision-only models via `timm`.
  - `VISION_LANGUAGE_BACKENDS` for CLIP-like backends:
    - Small: `clip_b32` (`ViT-B-32-quickgelu`), `clip_b16` (`ViT-B-16-quickgelu`), `siglip2` (`ViT-B-16-SigLIP2`), `mobileclip2` (`MobileCLIP2-S4`).
    - Medium: `openai` (`ViT-L-14-quickgelu`), `resnet50` (`RN50x64-quickgelu`), `coca` (`coca_ViT-L-14`).
    - Heavy: `openclipbase` (`ViT-H-14-quickgelu`), `vitg` (`ViT-bigG-14`), `eva02` (`EVA02-E-14-plus`), `convnext` (`convnext_xxlarge`).
    - Custom: `custom-clip`, `custom-clip-hf`.
  - `VISION_ONLY_BACKENDS` for timm vision-only backbones:
    - Small: `mobilenetv4` (`mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k`), `resnet18` (`resnet18.a1_in1k`), `efficientnet_b0` (`efficientnet_b0.ra_in1k`).
    - Medium: `dinov3` (`vit_large_patch16_dinov3.lvd1689m`).
    - Custom: `custom-vision`.
- Added an in-process cache so repeated calls to `load_backend` with the same `(backend, backend_model_id, device)` reuse the same model object.

### Benchmarking Utility

- Added `conformal_clip.benchmark.benchmark_models`:
  - Compares backends across calibration (`None`, `isotonic`, `sigmoid`) and conformal modes (`None`, `global`, `mondrian`) on fixed splits.
  - Returns two DataFrames (classification metrics and conformal metrics) plus styled variants with best values highlighted.
  - Uses the shared backend loader, CLIPWrapper, and conformal utilities for consistency.

### Resource Tiers

- Introduced a `resource_tier` argument to `benchmark_models`:
  - `resource_tier="low"` (default): only small backbones suitable for low-resource environments:
    - CLIP-like: `clip_b32`, `clip_b16`, `siglip2`, `mobileclip2`.
    - Vision-only: `mobilenetv4`, `resnet18`, `efficientnet_b0`.
  - `resource_tier="medium"`: low-tier plus mid-sized models that run comfortably on ~8–12 GB GPUs:
    - Adds: `openai`, `resnet50`, `coca`, `dinov3`.
  - `resource_tier="high"`: all non-custom backends, including the heaviest models:
    - Adds: `openclipbase`, `vitg`, `eva02`, `convnext`.
- When `backends` is provided explicitly, `resource_tier` is ignored and the user’s selection is used as-is.

### Documentation

- Updated README:
  - Extended the benchmark example to show `resource_tier="low"` and explain the low/medium/high tiers.
  - Added a clear resource warning that running benchmarks over many large backbones can require tens of GB of RAM and substantial disk cache space, and recommends restricting `backends` or using a lower `resource_tier` on constrained machines.

## Notes

- `open-clip-torch` is used without a strict version pin; helper functions such as `_pretrained_tags_by_model` make the backend loader robust to minor API changes.
- Existing public APIs (`few_shot_fault_classification_conformal`, `CLIPWrapper`, metrics, and visualization functions) are unchanged and remain backwards compatible.

