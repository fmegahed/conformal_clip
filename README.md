# conformal_clip

[![PyPI version](https://img.shields.io/pypi/v/conformal-clip.svg)](https://pypi.org/project/conformal-clip/)
[![Python versions](https://img.shields.io/pypi/pyversions/conformal-clip.svg)](https://pypi.org/project/conformal-clip/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Development Status](https://img.shields.io/badge/status-beta-orange.svg)](https://pypi.org/project/conformal-clip/)

> **⚠️ Beta Release**: This package is currently in beta (v0.1.1). The API is stable but may evolve based on user feedback. We welcome bug reports and feature requests via [GitHub Issues](https://github.com/fmegahed/conformal_clip/issues).

**Few-shot CLIP vision classification** with **conformal prediction** and **probability calibration** for manufacturing inspection and occupational safety applications.

---

## 🎯 Package Focus

This package focuses on **vision-only few-shot learning with CLIP** — specifically targeting scenarios where:

- ✅ **Vision encoder only**: We use CLIP's image encoder for few-shot classification based on exemplar images, **not** text prompts for test images.
- ✅ **No text captions**: Designed for applications like **manufacturing image inspection** and **occupational safety evaluations**, where images are captured automatically without associated captions or descriptions.
- ✅ **Conformal prediction**: Provides set-valued predictions with finite-sample coverage guarantees (both global and Mondrian/class-conditional).
- ✅ **Probability calibration**: Optional isotonic regression or Platt scaling (sigmoid) to improve probability estimates.
- ✅ **Comprehensive metrics**: Includes standard classification metrics and conformal set-specific metrics (coverage, set size, etc.).

**What we do NOT do:**
- ❌ We do NOT use CLIP's text encoder to generate text embeddings for test images.
- ❌ This is NOT a general-purpose CLIP library — it's specialized for vision-based few-shot learning in industrial/safety contexts.

---

## 🚀 Key Features
- **Zero-shot evaluation** with CLIP text prompts (for comparison/baseline).
- **Few-shot classification** using CLIP image exemplars only (vision encoder).
- **Conformal prediction** (Global and Mondrian) for set-valued predictions with finite-sample guarantees.
- **Optional probability calibration** (isotonic or sigmoid) before conformal scoring.
- **Comprehensive metrics** for both point predictions and conformal sets.
- **Simple I/O utilities** for local or URL-based image access and GitHub folder listings.
- **Visualizations**: confusion matrices and coverage summaries.

---

## 📂 Repository Layout
```
conformal_clip/
├── __init__.py                     # Public API exports
├── image_io.py                     # load_image, sample_urls
├── io_github.py                    # get_image_urls from GitHub folders
├── wrappers.py                     # encode_and_normalize, CLIPWrapper (sklearn-compatible)
├── zero_shot.py                    # evaluate_zero_shot_predictions
├── conformal.py                    # few_shot_fault_classification_conformal
├── metrics.py                      # compute_classification_metrics, compute_conformal_set_metrics, make_true_labels_from_counts
├── viz.py                          # plot_confusion_matrix helper
index.qmd                           # Quarto notebook demonstrating full workflow
index.html, index_files/            # Rendered notebook output
results/                            # Example experiment outputs (CSV + plots) - for reference only
data/                               # Example images
```

---

## ⚙️ Requirements
- **Python** ≥ 3.9  
- **Core packages** (see `requirements.txt`):  
  `jupyter`, `matplotlib`, `pandas`, `requests`, `scikit-learn`, `seaborn`  
- **CLIP** (OpenAI official):  
  ```bash
  pip install git+https://github.com/openai/CLIP.git
  ```
- **PyTorch** (with optional CUDA for GPU acceleration)

---

## 💻 Installation

### Prerequisites

**OpenAI CLIP** must be installed first:
```bash
pip install git+https://github.com/openai/CLIP.git
```

### Option 1: Core Package Only
Install just the conformal_clip library:
```bash
pip install conformal-clip
```

### Option 2: With Example Dataset
Install the package **with the example textile dataset**:
```bash
pip install "conformal-clip[data]"
```

**What does `[data]` add?**
The `[data]` extra installs the **conformal-clip-data** package, which includes:
- Simulated textile defect images (nominal, global defects, local defects)
- Used in the examples and reproducible workflow described in [Megahed et al., 2025](https://arxiv.org/pdf/2501.12596)
- Convenient functions to access image directories: `textile_simulated_root()`, `nominal_dir()`, `local_dir()`, `global_dir()`
- **Not required** if you're using your own images


---

## 📦 Dataset

If you installed with `[data]`, the dataset will be available automatically through the optional `conformal-clip-data` package.

These are the simulated textile images described in  **[Megahed et al., 2025](https://arxiv.org/pdf/2501.12596)** and  generated using the R package  [spc4sts](https://cran.r-project.org/web/packages/spc4sts/index.html).

Each image (250×250 px) was generated using ϕ₁ = 0.6 and ϕ₂ = 0.35 for nominal textures. Global defects were simulated by reducing both parameters by 5%, and  local defects were imposed via `spc4sts`’ defect-generation functions.

---

## 🧠 Usage Guide

This guide walks through the complete workflow for few-shot CLIP classification with conformal prediction.

---

### **Step 1: Load CLIP Model and Prepare Images**

```python
import os
import torch
import clip
import numpy as np
from pathlib import Path
from conformal_clip import load_image
from conformal_clip.image_io import sample_urls

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# ===================================================================
# Option A: From local paths (if using your own data)
# ===================================================================
# image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
# images = [load_image(p) for p in image_paths]
# preprocessed_images = [preprocess(img).unsqueeze(0).to(device) for img in images]

# ===================================================================
# Option B: If you installed with [data], use the example dataset
# ===================================================================
from conformal_clip_data import nominal_dir, local_dir, global_dir

# Helper function to list image files
exts = {"jpg", "jpeg", "png"}
def list_imgs(p: str):
    return [str(q) for q in Path(p).iterdir() if q.suffix.lower().lstrip(".") in exts]

# Get all image paths
nominal_paths = list_imgs(nominal_dir())
local_paths = list_imgs(local_dir())
global_paths = list_imgs(global_dir())

# Reproducible random sampling
rng = np.random.default_rng(2024)

# Sample test set: 100 images (50 nominal, 25 local, 25 global)
test_nominal_paths, remaining_nominal_paths = sample_urls(nominal_paths, 50, rng)
test_global_paths, remaining_global_paths = sample_urls(global_paths, 25, rng)
test_local_paths, remaining_local_paths = sample_urls(local_paths, 25, rng)

test_defective_paths = test_global_paths + test_local_paths
test_paths = test_nominal_paths + test_defective_paths
test_image_filenames = [os.path.basename(p) for p in test_paths]
```

---

### **Step 2: Create Exemplar Banks (Few-Shot Learning)**

For few-shot learning, sample exemplar images from each class:

```python
# Sample training (few-shot) exemplars: 50 per class
train_nominal_paths, remaining_nominal_paths = sample_urls(remaining_nominal_paths, 50, rng)
train_global_paths, remaining_global_paths = sample_urls(remaining_global_paths, 25, rng)
train_local_paths, remaining_local_paths = sample_urls(remaining_local_paths, 25, rng)
train_defective_paths = train_global_paths + train_local_paths

# Create descriptions for few-shot references (used for traceability only)
nominal_descriptions = [
    f"Image {os.path.basename(p)}: nominal textile, consistent weave, no visible defects."
    for p in train_nominal_paths
]
global_descriptions = [
    f"Image {os.path.basename(p)}: global distortion, uniform shift across texture."
    for p in train_global_paths
]
local_descriptions = [
    f"Image {os.path.basename(p)}: localized defect disrupting weave pattern."
    for p in train_local_paths
]
defective_descriptions = global_descriptions + local_descriptions

# Load and preprocess exemplar images
nominal_images = [preprocess(load_image(p)).unsqueeze(0).to(device)
                  for p in train_nominal_paths]
defective_images = [preprocess(load_image(p)).unsqueeze(0).to(device)
                    for p in train_defective_paths]
```

**Note**: Descriptions are optional and used only for traceability/logging, not for text encoding.

---

### **Step 3: Prepare Calibration and Test Sets**

```python
# Calibration set (same size as training: 50 nominal, 25 global, 25 local)
cal_nominal_paths, remaining_nominal_paths = sample_urls(remaining_nominal_paths, 50, rng)
cal_global_paths, remaining_global_paths = sample_urls(remaining_global_paths, 25, rng)
cal_local_paths, remaining_local_paths = sample_urls(remaining_local_paths, 25, rng)
cal_defective_paths = cal_global_paths + cal_local_paths

# Load and preprocess calibration images
calib_nominal_images = [preprocess(load_image(p)).unsqueeze(0).to(device)
                        for p in cal_nominal_paths]
calib_defective_images = [preprocess(load_image(p)).unsqueeze(0).to(device)
                          for p in cal_defective_paths]
calib_images = calib_nominal_images + calib_defective_images
calib_labels = (["Nominal"] * len(calib_nominal_images) +
                ["Defective"] * len(calib_defective_images))

# Load and preprocess test images (already sampled in Step 1)
test_nominal_images = [preprocess(load_image(p)).unsqueeze(0).to(device)
                       for p in test_nominal_paths]
test_defective_images = [preprocess(load_image(p)).unsqueeze(0).to(device)
                         for p in test_defective_paths]
test_images = test_nominal_images + test_defective_images

# Bookkeeping for metrics
labels = ["Nominal", "Defective"]
label_counts = [len(test_nominal_images), len(test_defective_images)]

print(
    f"Train: Nominal={len(nominal_images)}, Defective={len(defective_images)} | "
    f"Calib: Nominal={len(calib_nominal_images)}, Defective={len(calib_defective_images)} | "
    f"Test: Nominal={len(test_nominal_images)}, Defective={len(test_defective_images)}"
)
```

---

### **Step 4: Run Few-Shot Classification with Conformal Prediction**

```python
from conformal_clip import few_shot_fault_classification_conformal

# Create results directory (optional - files will be saved here)
import os
os.makedirs("results", exist_ok=True)

results = few_shot_fault_classification_conformal(
    model=model,
    test_images=test_images,
    test_image_filenames=test_image_filenames,
    nominal_images=nominal_images,
    nominal_descriptions=nominal_descriptions,
    defective_images=defective_images,
    defective_descriptions=defective_descriptions,
    calib_images=calib_images,
    calib_labels=calib_labels,
    alpha=0.1,                      # 90% coverage target
    temperature=1.0,                # Softmax temperature
    mondrian=True,                  # Per-class coverage (recommended)
    prob_calibration="isotonic",    # Calibrate probabilities (test both "isotonic" and "sigmoid" on your data)
    allow_empty=False,              # Force at least one label in prediction set
    csv_path="results",             # Directory for output files
    csv_filename="exp_results_conformal.csv",
    print_one_liner=True            # Print results for each test image
)
```

**Output**: CSV file saved to `results/exp_results_conformal.csv` with columns:
```
datetime_of_operation, alpha, temperature, mondrian, image_path, image_name,
point_prediction, prediction_set, set_size, Nominal_prob, Defective_prob,
nominal_description, defective_description
```

---

### **Step 5: Compute Classification Metrics**

```python
from conformal_clip import compute_classification_metrics, compute_conformal_set_metrics

# Standard point prediction metrics (labels and label_counts already defined in Step 3)
metrics_df = compute_classification_metrics(
    csv_file="results/exp_results_conformal.csv",
    labels=labels,
    label_counts=label_counts,
    save_confusion_matrix=True,
    cm_file_path="results",
    cm_file_name="confusion_matrix.png",
    cm_title="Few-Shot CLIP Classification"
)
print(metrics_df)
# Output: Accuracy, Sensitivity (Recall), Specificity, Precision, F1 Score, AUC

# Conformal set-specific metrics
conformal_metrics_df = compute_conformal_set_metrics(
    csv_file="results/exp_results_conformal.csv",
    labels=labels,
    label_counts=label_counts
)
print(conformal_metrics_df)
# Output: Coverage (overall and per-class)
```

---

## 🧩 Important Notes

### **Preprocessing**
Always use CLIP's provided `preprocess` before passing images to the model:
```python
model, preprocess = clip.load("ViT-L/14", device=device)
preprocessed_image = preprocess(pil_image).unsqueeze(0).to(device)
```

### **Probability Calibration**
Choose between isotonic regression and sigmoid (Platt) scaling:
- **`"isotonic"`** — According to [sklearn documentation](https://scikit-learn.org/stable/modules/calibration.html#isotonic), isotonic regression is non-parametric and preserves monotonicity but can be prone to overfitting on small calibration sets.
- **`"sigmoid"`** — According to [sklearn documentation](https://scikit-learn.org/stable/modules/calibration.html#sigmoid), sigmoid (Platt) scaling uses logistic regression and may be more robust for smaller calibration sets.

**Recommendation**: Test both methods on your data. In our experiments with 100 calibration samples (50 per class) on textile defect images, isotonic regression performed better. Your results may vary depending on your dataset characteristics.

### **Conformal Prediction Modes**
- **Mondrian** (`mondrian=True`) — Per-class coverage guarantees. Recommended when class balance matters.
- **Global** (`mondrian=False`) — Overall coverage guarantee across all classes.

### **Allow Empty Sets**
- **`allow_empty=False`** (default) — Forces at least one label (uses argmax if conformal set is empty).
- **`allow_empty=True`** — Allows abstention (empty prediction set) when model is uncertain.

### **Temperature Scaling**
The `temperature` parameter controls softmax sharpness:
- `temperature > 1.0` → softer probabilities (less confident)
- `temperature < 1.0` → sharper probabilities (more confident)
- `temperature = 1.0` → standard softmax (default)  

---

## 📚 Citation

If you use this package in your research, please cite:

```bibtex
@misc{megahed2025adaptingopenaisclipmodel,
      title={Adapting OpenAI's CLIP Model for Few-Shot Image Inspection in Manufacturing Quality Control: An Expository Case Study with Multiple Application Examples}, 
      author={Fadel M. Megahed and Ying-Ju Chen and Bianca Maria Colosimo and Marco Luigi Giuseppe Grasso and L. Allison Jones-Farmer and Sven Knoth and Hongyue Sun and Inez Zwetsloot},
      year={2025},
      eprint={2501.12596},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.12596}, 
}
```

**Paper**: [Megahed et al., 2025 - arXiv:2501.12596](https://arxiv.org/abs/2501.12596)

---

## ⚖️ License
This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.