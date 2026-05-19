# conformal_clip

[![PyPI version](https://img.shields.io/pypi/v/conformal-clip.svg)](https://pypi.org/project/conformal-clip/)
[![Python versions](https://img.shields.io/pypi/pyversions/conformal-clip.svg)](https://pypi.org/project/conformal-clip/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Development Status](https://img.shields.io/badge/status-beta-orange.svg)](https://pypi.org/project/conformal-clip/)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://fmegahed.github.io/research/conformal_clip/)

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

## Setup

All Quickstart blocks below build on a single setup flow: pick a device, load a backend, gather image paths, and split them into few-shot banks, a calibration set, and a test set. Run Step&nbsp;1, then **either Step&nbsp;2A or Step&nbsp;2B**, then Step&nbsp;3. After that you can run any Quickstart block.

### Step 1 — Device + backend
```python
import torch
from conformal_clip import load_backend

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess_fn, tokenize_fn = load_backend("openclipbase", None, device)
```

### Step 2 — Image paths (pick A or B)

Both options produce the same six path lists (`nom_train`, `nom_calib`, `nom_test`, `def_train`, `def_calib`, `def_test`) that Step&nbsp;3 consumes.

**Option A — example textile dataset (installed via `conformal-clip[data]`)**
```python
import os
from conformal_clip_data import nominal_dir, local_dir, global_dir

def list_paths(d):
    exts = {".jpg", ".jpeg", ".png"}
    return [os.path.join(d, f) for f in os.listdir(d) if os.path.splitext(f)[1].lower() in exts]

nominal_paths = list_paths(nominal_dir())
local_paths   = list_paths(local_dir())
global_paths  = list_paths(global_dir())

nom_train, nom_calib, nom_test = nominal_paths[:50], nominal_paths[50:100], nominal_paths[100:150]
def_train = local_paths[:25]   + global_paths[:25]
def_calib = local_paths[25:50] + global_paths[25:50]
def_test  = local_paths[50:75] + global_paths[50:75]
```

**Option B — your own local folders**

Expects a `nominal/` and a `defective/` directory; adjust slice sizes to your dataset.
```python
import os

base_dir      = "./data/textile_images/simulated"
nominal_dir   = os.path.join(base_dir, "nominal")
defective_dir = os.path.join(base_dir, "defective")

def list_paths(d):
    exts = {".jpg", ".jpeg", ".png"}
    return [os.path.join(d, f) for f in os.listdir(d) if os.path.splitext(f)[1].lower() in exts]

nominal_paths   = list_paths(nominal_dir)
defective_paths = list_paths(defective_dir)

nom_train, nom_calib, nom_test = nominal_paths[:50],   nominal_paths[50:100],   nominal_paths[100:150]
def_train, def_calib, def_test = defective_paths[:50], defective_paths[50:100], defective_paths[100:150]
```

### Step 3 — Build PIL banks, labels, and filenames

This block runs after either Option A or Option B.
```python
import os
from PIL import Image

pil_nom_bank = [Image.open(p).convert("RGB") for p in nom_train]
pil_def_bank = [Image.open(p).convert("RGB") for p in def_train]
pil_calib    = [Image.open(p).convert("RGB") for p in (nom_calib + def_calib)]
pil_test     = [Image.open(p).convert("RGB") for p in (nom_test  + def_test)]

calib_labels   = ["Nominal"] * len(nom_calib) + ["Defective"] * len(def_calib)
test_labels    = ["Nominal"] * len(nom_test)  + ["Defective"] * len(def_test)
test_filenames = [os.path.basename(p) for p in (nom_test + def_test)]
```

> **PIL images vs. preprocessed tensors.** The APIs in this package differ in the form they expect:
> - `benchmark_models` consumes **PIL images** directly and applies each backend's preprocessing internally as it sweeps.
> - `few_shot_fault_classification_conformal` and `evaluate_zero_shot_predictions` consume **preprocessed tensors** — pass PIL images through `preprocess_fn` from `load_backend(...)` first.
>
> Setup ends with PIL banks (`pil_*`). Each Quickstart block below applies `preprocess_fn` when (and only when) it's needed.

---

## Quickstart

### Few-shot classification with conformal prediction

_Continues from Setup above._ Few-shot conformal expects preprocessed tensors, so we first run each PIL bank through `preprocess_fn` (returned by `load_backend` in Step&nbsp;1).

```python
from conformal_clip import few_shot_fault_classification_conformal

nominal_images   = [preprocess_fn(img) for img in pil_nom_bank]
defective_images = [preprocess_fn(img) for img in pil_def_bank]
calib_images     = [preprocess_fn(img) for img in pil_calib]
test_images      = [preprocess_fn(img) for img in pil_test]

results = few_shot_fault_classification_conformal(
    model=model,
    test_images=test_images,
    test_image_filenames=test_filenames,
    nominal_images=nominal_images,
    nominal_descriptions=["nominal textile sample"] * len(nominal_images),
    defective_images=defective_images,
    defective_descriptions=["defective textile sample"] * len(defective_images),
    calib_images=calib_images,
    calib_labels=calib_labels,
    alpha=0.1,
    mondrian=True,
    prob_calibration="isotonic",  # or "sigmoid" or None
)
```

### Zero-shot classification (CLIP-like backends only)

_Continues from Setup above._ Zero-shot also expects preprocessed tensors and additionally needs `tokenize_fn`. Setup loaded `openclipbase` (a CLIP-like backend), so `tokenize_fn` is non-`None`.

```python
from conformal_clip import evaluate_zero_shot_predictions

test_images = [preprocess_fn(img) for img in pil_test]

metrics_df, results_df = evaluate_zero_shot_predictions(
    labels=["Nominal", "Defective"],
    label_counts=[test_labels.count("Nominal"), test_labels.count("Defective")],
    test_images=test_images,
    test_image_filenames=test_filenames,
    model=model,
    device=device,
    tokenize_fn=tokenize_fn,
    save_confusion_matrix=True,
)
```

### Benchmark across backends, calibration, and conformal settings

_Continues from Setup above._ `benchmark_models` takes **PIL images** directly — no `preprocess_fn` step is needed, since the function loads each backend internally and applies that backend's own preprocessing as it sweeps.

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

## Examples

The `examples/` folder contains runnable end-to-end scripts. Four manufacturing-inspection datasets are demonstrated: **textile** (the default, ships via `conformal-clip[data]`), **extrusion**, **microstructure**, and **pipe**.

Per-backend few-shot scripts (each runs one backend on the textile dataset and shows the full pipeline — load, split, calibrate, conformal):
- `textile_mobileclip2.py` — MobileCLIP2 (lightweight CLIP variant)
- `textile_mobilenetv4.py` — MobileNetV4 (vision-only timm)
- `zero_shot_openclip.py` — Zero-shot baseline with an OpenCLIP backbone

Benchmark suites (sweep multiple backends, calibration methods, and conformal modes; emit CSV + styled HTML):
- `benchmark_textile.py`, `benchmark_extrusion.py`, `benchmark_microstructure.py`, `benchmark_pipe.py`

Discovery and customization:
- `custom_openclip_example.py` — using `custom-clip` and `custom-clip-hf`
- `list_models_openclip_timm.py` — browse available OpenCLIP and timm models

Each `_shared_<dataset>.py` module (e.g., `_shared_textile.py`) provides dataset-specific path resolution and reproducible train/calibration/test splits — the same pattern you can copy when adapting these scripts to your own data.

---

## Discover and use custom models

Beyond the named backends in the tables above, you can load any OpenCLIP-compatible model or any timm vision backbone using the `custom-clip`, `custom-clip-hf`, or `custom-vision` backends. The snippets in this section are standalone — they do not depend on the Setup flow.

### Custom OpenCLIP (vision–language) models

`custom-clip` accepts an OpenCLIP built-in model name (optionally with a pretrained tag via `"model@tag"`). `custom-clip-hf` accepts a Hugging Face repo id (`"hf-hub:org/repo"`). Browse models: [Hugging Face OpenCLIP](https://huggingface.co/models?library=open_clip).

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
import torch
from conformal_clip import load_backend

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using just the model name (the loader picks a sensible pretrained tag)
model, preprocess_fn, tokenize_fn = load_backend(
    backend="custom-clip", backend_model_id="ViT-L-14-quickgelu", device=device,
)

# Or pin a specific tag with "model@tag"
model, preprocess_fn, tokenize_fn = load_backend(
    backend="custom-clip", backend_model_id="ViT-L-14-quickgelu@openai", device=device,
)
```

Load from Hugging Face (OpenCLIP-compatible weights) using `custom-clip-hf`:
```python
# Gated repos require HF_TOKEN in the environment or a .env file (see Environment Setup).
model, preprocess_fn, tokenize_fn = load_backend(
    backend="custom-clip-hf", backend_model_id="hf-hub:org-or-user/repo-id", device=device,
)
```

### Custom timm (vision-only) models

`custom-vision` accepts any timm model id. The returned `tokenize_fn` is `None` because timm backbones do not have a text encoder, so vision-only backends are usable with `few_shot_fault_classification_conformal` and `benchmark_models` but **not** with `evaluate_zero_shot_predictions`. Browse models: [Hugging Face timm](https://huggingface.co/models?library=timm).

List timm model names programmatically:
```python
import timm

# All models that have pretrained weights available
names = timm.list_models(pretrained=True)
print(f"Found {len(names)} pretrained timm models")
print("First 20:", names[:20])

# Filter by family/pattern
print("mobilenet*:", timm.list_models("mobilenet*", pretrained=True)[:10])
print("convnext*:", timm.list_models("convnext*", pretrained=True)[:10])
```

Load a custom timm model:
```python
import torch
from conformal_clip import load_backend

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, preprocess_fn, tokenize_fn = load_backend(
    backend="custom-vision",
    backend_model_id="convnext_base.fb_in22k_ft_in1k",
    device=device,
)
assert tokenize_fn is None  # vision-only backbones have no text encoder
```

---

## Project structure

```
conformal_clip/
├── conformal_clip/          # Main package
│   ├── __init__.py          # Public API surface (see __all__)
│   ├── backends.py          # load_backend(...) for OpenCLIP and timm
│   ├── wrappers.py          # CLIPWrapper: sklearn-compatible few-shot classifier
│   ├── conformal.py         # Global and Mondrian conformal prediction
│   ├── zero_shot.py         # Zero-shot evaluation for CLIP-like models
│   ├── metrics.py           # Classification and conformal-set metrics
│   ├── benchmark.py         # benchmark_models(...) sweep utility
│   ├── viz.py               # Confusion matrix plotting
│   ├── image_io.py          # Image loading from disk or URL
│   ├── io_github.py         # List image URLs from GitHub directories
│   └── data_utils.py        # Train/calibration/test split helpers
├── examples/                # Runnable end-to-end scripts (see "Examples" above)
└── tests/                   # Unit and integration tests (pytest)
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
