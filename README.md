# conformal_clip

[![PyPI version](https://img.shields.io/pypi/v/conformal-clip.svg)](https://pypi.org/project/conformal-clip/)
[![Python versions](https://img.shields.io/pypi/pyversions/conformal-clip.svg)](https://pypi.org/project/conformal-clip/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Development Status](https://img.shields.io/badge/status-beta-orange.svg)](https://pypi.org/project/conformal-clip/)

Few-shot vision classification with conformal prediction and optional probability calibration.

This package runs CLIP-like models via [open-clip-torch](https://github.com/mlfoundations/open_clip) (any model in [Hugging Face OpenCLIP](https://huggingface.co/models?library=open_clip)) and also supports timm vision-only models for few-shot conformal prediction ([Hugging Face timm models](https://huggingface.co/models?library=timm)).

---

## Features
- Backends: OpenCLIP (CLIP-like, vision+text) and timm (vision-only)
- Few-shot classification using exemplar image banks
- Conformal prediction (global and Mondrian) with finite-sample coverage
- Optional calibration (isotonic or sigmoid/Platt)
- Zero-shot baseline for CLIP-like models
- Benchmark utility to compare backends and settings across the same splits

### Backends overview

Backends are provided in two broad categories: CLIP-like vision–language encoders (via `open-clip-torch`) and image-only encoders (via `timm`). All of them can be loaded with `load_backend(backend, backend_model_id, device)`; the recommendations below are approximate and assume small batch sizes and fp16 where possible.

**CLIP-like (vision–language) encoders**

| Family        | Backend key   | OpenCLIP model id         | Recommended environment           | Notes |
|--------------|---------------|---------------------------|------------------------------------|-------|
| ViT (small)  | `clip_b32`    | `ViT-B-32-quickgelu`      | Low–medium (4–8 GB GPU or CPU)    | Lightest ViT CLIP; good starting point when resources are tight. |
| ViT (small)  | `clip_b16`    | `ViT-B-16-quickgelu`      | Medium (≥8 GB GPU)                | More accurate than B/32 at modest extra cost. |
| ViT (base)   | `siglip2`     | `ViT-B-16-SigLIP2`        | Medium (≥8 GB GPU)                | ViT-B model with SigLIP2 loss; strong trade-off between accuracy and cost. |
| ViT (mobile) | `mobileclip2` | `MobileCLIP2-S4`          | Low (CPU, 4–8 GB GPU, edge)       | Mobile-optimized CLIP; preferred for low-power or edge deployments. |
| ViT (large)  | `openai`      | `ViT-L-14-quickgelu`      | Medium–high (≥8–12 GB GPU)        | Classic CLIP baseline; strong general performance. |
| ViT (xlarge) | `openclipbase`| `ViT-H-14-quickgelu`      | High (≥16 GB GPU)                 | Larger ViT-H encoder; use when memory is ample. |
| ViT (giant)  | `vitg`        | `ViT-bigG-14`             | Very high (≥24 GB GPU)            | Extremely large model; for offline or benchmark use only. |
| ResNet CLIP  | `resnet50`    | `RN50x64-quickgelu`       | Medium (≥8 GB GPU or strong CPU)  | Deep CNN CLIP; useful as a non-ViT baseline. |
| EVA family   | `eva02`       | `EVA02-E-14-plus`         | High (≥16–24 GB GPU)              | High-capacity ViT-style model; heavy but strong. |
| ConvNeXt     | `convnext`    | `convnext_xxlarge`        | Very high (≥24 GB GPU)            | Very large ConvNeXt CLIP; avoid on small GPUs. |
| CoCa         | `coca`        | `coca_ViT-L-14`           | High (≥12–16 GB GPU)              | Captioning-oriented CLIP variant; strong but memory-hungry. |

**Image-only (vision encoders via timm)**

| Family       | Backend key      | timm model id                                   | Recommended environment        | Notes |
|-------------|------------------|-------------------------------------------------|---------------------------------|-------|
| Lightweight | `mobilenetv4`    | `mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k` | Low (CPU, 4–8 GB GPU)       | Very efficient mobile-style CNN; best when resources are tight. |
| Lightweight | `resnet18`       | `resnet18.a1_in1k`                              | Low (CPU, 4–8 GB GPU)          | Classic small ResNet; easy to run and debug. |
| Lightweight | `efficientnet_b0`| `efficientnet_b0.ra_in1k`                       | Low–medium (CPU, 4–8 GB GPU)   | Strong accuracy/efficiency balance among small CNNs. |
| ViT-L       | `dinov3`         | `vit_large_patch16_dinov3.lvd1689m`             | Medium–high (≥8–16 GB GPU)     | Self-supervised ViT-Large; heavy but strong general-purpose features. |

---

## Install

Core package:
```
pip install conformal-clip
```

With example dataset (textile images):
```
pip install "conformal-clip[data]"
```

Notes:
- Set `HF_TOKEN` in your environment (or a `.env` file) if you need access to gated models (e.g., DINOv3). The loader forwards it to `HUGGINGFACE_HUB_TOKEN`.
- PyTorch with CUDA is recommended for speed but not required.

---

## Environment Setup

- Hugging Face token (for gated repos like some DINOv3 builds):
  - In shell: `export HF_TOKEN=hf_...` (Linux/macOS) or `set HF_TOKEN=hf_...` (Windows)
  - Or create a `.env` file next to your script with `HF_TOKEN=hf_...`.
  - The loader maps `HF_TOKEN` to `HUGGINGFACE_HUB_TOKEN` automatically.
- CUDA (optional): Install a CUDA-enabled PyTorch build from [PyTorch Get Started (Locally)](https://pytorch.org/get-started/locally/) then use `device = torch.device("cuda")`.

---

## Quickstart

Load a backend and run few-shot + conformal:
```python
import torch
from conformal_clip import load_backend, few_shot_fault_classification_conformal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess_fn, _ = load_backend("openclipbase", None, device)

# Prepare PIL -> tensor using preprocess_fn, then call few_shot_fault_classification_conformal
# nominal_images, defective_images, calib_images, test_images: lists of preprocessed tensors
# calib_labels: list[str] of ground-truth for calibration
results = few_shot_fault_classification_conformal(
    model=model,
    test_images=test_images,
    test_image_filenames=test_filenames,
    nominal_images=nominal_images,
    nominal_descriptions=["..."] * len(nominal_images),
    defective_images=defective_images,
    defective_descriptions=["..."] * len(defective_images),
    calib_images=calib_images,
    calib_labels=calib_labels,
    alpha=0.1,
    mondrian=True,
    prob_calibration="isotonic",  # or "sigmoid" or None
)
```

Zero-shot (CLIP-like backends only):
```python
from conformal_clip import evaluate_zero_shot_predictions

model, preprocess_fn, tokenize_fn = load_backend("openai", None, device)
metrics_df, results_df = evaluate_zero_shot_predictions(
    labels=my_labels,
    label_counts=my_label_counts,
    test_images=[preprocess_fn(pil) for pil in my_pil_images],
    test_image_filenames=my_filenames,
    model=model,
    device=device,
    tokenize_fn=tokenize_fn,
    save_confusion_matrix=True,
)
```

Benchmark across backends, calibration and conformal settings:
```python
from conformal_clip import benchmark_models

cls_df, cp_df, cls_style, cp_style = benchmark_models(
    train_nominal_images=pil_nom_bank,
    train_defective_images=pil_def_bank,
    calib_images=pil_calib,
    calib_labels=calib_labels,
    test_images=pil_test,
    test_labels=test_labels,
    device=device,
    seed=2025,
    # resource_tier controls which backends are run when backends is None.
    # Defaults to "low" (small models suitable for low-resource environments).
    resource_tier="low",  # or "medium" or "high"
    calibration_methods=(None, "isotonic", "sigmoid"),
    conformal_modes=(None, "global", "mondrian"),
    alpha_list=(0.1,),
)
```

**Resource warning:** many CLIP/timm backbones are large (hundreds of MB to multiple GB per model). Running `benchmark_models` over many backends can require tens of gigabytes of RAM and substantial disk cache space. We do not recommend running the full benchmark configuration on machines with limited memory or storage; instead, restrict the `backends` argument to a small subset of models.

---

## Prepare Images

Using the example textile dataset (installed via `conformal-clip[data]`):
```python
import os
from PIL import Image
from conformal_clip_data import nominal_dir, local_dir, global_dir

def list_paths(d):
    exts = {".jpg", ".jpeg", ".png"}
    return [os.path.join(d, f) for f in os.listdir(d) if os.path.splitext(f)[1].lower() in exts]

nominal_paths = list_paths(nominal_dir())
local_paths = list_paths(local_dir())
global_paths = list_paths(global_dir())

# Few-shot banks (exemplars)
pil_nom_bank = [Image.open(p).convert("RGB") for p in nominal_paths[:50]]
pil_def_bank = [Image.open(p).convert("RGB") for p in (local_paths[:25] + global_paths[:25])]

# Calibration and test (example split)
pil_calib = [Image.open(p).convert("RGB") for p in (nominal_paths[50:100] + local_paths[25:50] + global_paths[25:50])]
calib_labels = ["Nominal"] * 50 + ["Defective"] * 50
pil_test = [Image.open(p).convert("RGB") for p in (nominal_paths[100:150] + local_paths[50:75] + global_paths[50:75])]
test_labels = ["Nominal"] * 50 + ["Defective"] * 50

# Preprocess to tensors for a given backend
# model, preprocess_fn, _ = load_backend("openclipbase", None, device)
nominal_images = [preprocess_fn(img) for img in pil_nom_bank]
defective_images = [preprocess_fn(img) for img in pil_def_bank]
calib_images = [preprocess_fn(img) for img in pil_calib]
test_images = [preprocess_fn(img) for img in pil_test]
```

Using your own local folders (commented template):
```python
# import os
# from PIL import Image
#
# base_dir = "./data/textile_images/simulated"
# nominal_dir = os.path.join(base_dir, "nominal")
# defective_dir = os.path.join(base_dir, "defective")
#
# def list_paths(d):
#     exts = {".jpg", ".jpeg", ".png"}
#     return [os.path.join(d, f) for f in os.listdir(d) if os.path.splitext(f)[1].lower() in exts]
#
# nominal_paths = list_paths(nominal_dir)
# defective_paths = list_paths(defective_dir)
#
# pil_nom_bank = [Image.open(p).convert("RGB") for p in nominal_paths[:50]]
# pil_def_bank = [Image.open(p).convert("RGB") for p in defective_paths[:50]]
# pil_calib = [Image.open(p).convert("RGB") for p in (nominal_paths[50:100] + defective_paths[50:100])]
# calib_labels = ["Nominal"] * 50 + ["Defective"] * 50
# pil_test = [Image.open(p).convert("RGB") for p in (nominal_paths[100:150] + defective_paths[100:150])]
# test_labels = ["Nominal"] * 50 + ["Defective"] * 50
#
# model, preprocess_fn, _ = load_backend("openclipbase", None, device)
# nominal_images = [preprocess_fn(img) for img in pil_nom_bank]
# defective_images = [preprocess_fn(img) for img in pil_def_bank]
# calib_images = [preprocess_fn(img) for img in pil_calib]
# test_images = [preprocess_fn(img) for img in pil_test]
```

---

## Examples

See the `examples/` folder:
- `textile_*.py`: Run each backend on the textile dataset (installed via `conformal-clip[data]`). Each script also shows how to adapt to your own data (commented snippet).
- `benchmark_textile.py`: End-to-end benchmark and HTML output with highlighted best metrics.

---

## Discover and Use Custom OpenCLIP Models

You can list available model names (and their pretrained tags) directly from [`open_clip`](https://github.com/mlfoundations/open_clip). These names can be used with the `custom-clip` backend (and you may optionally provide a specific tag via the `model@tag` format). For Hugging Face repo IDs in the OpenCLIP ecosystem, use `custom-clip-hf`.

Browse OpenCLIP models on Hugging Face: [Hugging Face OpenCLIP models](https://huggingface.co/models?library=open_clip)

List available built-in model names and their tags:
```python
import open_clip

# Dict[str, Set[str]] mapping model_name -> available pretrained tags
by_model = open_clip.list_pretrained_tags_by_model()

print("Some available model names (built-ins):")
for name in sorted(by_model.keys())[:20]:
    print(" ", name)

# Inspect tags for a specific model
model_name = "ViT-L-14-quickgelu"
print("Available pretrained tags for", model_name, ":", sorted(by_model.get(model_name, [])))
```

Load a custom built-in model (optionally specify a tag):
```python
from conformal_clip import load_backend
device = "cuda"  # or "cpu"

# Using just the model name (the loader will pick a sensible pretrained tag)
model, preprocess_fn, tokenize_fn = load_backend(
    backend="custom-clip", backend_model_id="ViT-L-14-quickgelu", device=device
)

# Or explicitly specify a tag using the "model@tag" form
model, preprocess_fn, tokenize_fn = load_backend(
    backend="custom-clip", backend_model_id="ViT-L-14-quickgelu@openai", device=device
)
```

Load from Hugging Face (OpenCLIP-compatible weights) using `custom-clip-hf`:
```python
# Example: an hf-hub repo id (ensure you have access if it’s gated)
model, preprocess_fn, tokenize_fn = load_backend(
    backend="custom-clip-hf", backend_model_id="hf-hub:org-or-user/repo-id", device=device
)
```

---

Browse timm models on Hugging Face: [Hugging Face timm models](https://huggingface.co/models?library=timm)

List timm model names programmatically (vision-only backends):
```python
import timm

# All models that have pretrained weights available
names = timm.list_models(pretrained=True)
print(f"Found {len(names)} pretrained timm models")
print("First 20:", names[:20])

# Filter by family/pattern (e.g., mobilenet*, convnext*)
print("mobilenet*:", timm.list_models("mobilenet*", pretrained=True)[:10])
print("convnext*:", timm.list_models("convnext*", pretrained=True)[:10])
```

---

## Project Structure

```
conformal_clip/
├── conformal_clip/          # Main package
│   ├── __init__.py          # Package initialization and public API
│   ├── backends.py          # Model loading for OpenCLIP and timm backends
│   ├── wrappers.py          # CLIPWrapper: sklearn-compatible classifier
│   ├── conformal.py         # Conformal prediction (global and Mondrian)
│   ├── zero_shot.py         # Zero-shot evaluation for CLIP-like models
│   ├── metrics.py           # Classification and conformal metrics
│   ├── benchmark.py         # Systematic benchmarking across backends
│   └── viz.py               # Visualization utilities (confusion matrices)
├── examples/                # Example scripts
│   ├── textile_mobileclip2.py      # Few-shot with MobileCLIP2
│   ├── textile_mobilenetv4.py      # Few-shot with MobileNetV4 (timm)
│   ├── zero_shot_openclip.py       # Zero-shot classification example
│   ├── benchmark_textile.py        # Full benchmarking suite
│   ├── custom_openclip_example.py  # Using custom OpenCLIP models
│   ├── list_models_openclip_timm.py # Discover available models
│   └── _shared_textile.py          # Shared data loading utilities
└── tests/                   # Unit and integration tests
```

---

## Citation

If you use this package in your research, please cite:

```bibtex
@misc{megahed2025adaptingopenaisclipmodel,
  title={Adapting OpenAI's CLIP Model for Few-Shot Image Inspection in Manufacturing Quality Control: An Expository Case Study with Multiple Application Examples},
  author={Fadel M. Megahed and Ying-Ju Chen and Bianca Maria Colosimo and Marco Luigi Giuseppe Grasso and L. Allison Jones-Farmer and Sven Knoth and Hongyue Sun and Inez Zwetsloot},
  year={2025},
  eprint={2501.12596},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2501.12596}
}
```

---

## License
MIT License (see `LICENSE`).
