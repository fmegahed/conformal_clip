# Release v0.1.1 - Initial Public Release

We're excited to announce the first public release of **conformal_clip** on PyPI! 🎉

## 🎯 Overview

`conformal_clip` is a Python package for **vision-only few-shot learning with CLIP**, featuring conformal prediction and probability calibration. Designed for manufacturing inspection and occupational safety applications where images are captured automatically without text captions.

## ✨ Features

### Core Functionality
- **Few-shot CLIP classification** using vision encoder only (no text encoding for test images)
- **Conformal prediction** with finite-sample coverage guarantees:
  - Global conformal prediction (overall coverage)
  - Mondrian conformal prediction (per-class coverage)
- **Probability calibration** via isotonic regression or sigmoid (Platt) scaling
- **Comprehensive metrics** for both point predictions and conformal sets

### Tools & Utilities
- Zero-shot evaluation baseline for comparison
- Image I/O utilities supporting local files and URLs
- GitHub folder image fetching
- Confusion matrix visualization
- CSV output with detailed per-image results

### Documentation & Examples
- Comprehensive README with step-by-step usage guide
- Three complete example scripts:
  - `basic_usage.py` - Minimal working example
  - `textile_inspection.py` - Full workflow with multiple configurations
  - `custom_dataset.py` - Adapting to your own dataset
- Detailed docstrings with examples for all public functions

### Optional Dataset
- Simulated textile defect images available via `conformal-clip[data]`
- Described in [Megahed et al., 2025](https://arxiv.org/abs/2501.12596)

## 🚀 Installation

**Prerequisites**: Install OpenAI CLIP first
```bash
pip install git+https://github.com/openai/CLIP.git
```

**Install conformal_clip**:
```bash
# Core package only
pip install conformal-clip

# With example textile dataset
pip install "conformal-clip[data]"
```

## 📖 Quick Start

```python
import torch
import clip
from conformal_clip import (
    few_shot_fault_classification_conformal,
    compute_classification_metrics,
    compute_conformal_set_metrics,
)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# Prepare your data: exemplar banks, calibration set, test set
# ... (see examples/ directory for complete workflows)

# Run conformal prediction
results = few_shot_fault_classification_conformal(
    model=model,
    test_images=test_images,
    test_image_filenames=test_filenames,
    nominal_images=nominal_bank,
    defective_images=defective_bank,
    calib_images=calib_images,
    calib_labels=calib_labels,
    alpha=0.1,                      # 90% coverage target
    mondrian=True,                  # Per-class coverage
    prob_calibration="isotonic",    # Calibrate probabilities
    csv_path="results",
)

# Compute metrics
metrics = compute_classification_metrics(...)
conformal_metrics = compute_conformal_set_metrics(...)
```

## 🔧 Technical Details

### Requirements
- Python ≥ 3.9
- PyTorch
- OpenAI CLIP (installed separately)
- Standard scientific Python stack (NumPy, pandas, scikit-learn, matplotlib, seaborn)

### Key Improvements in v0.1.1
- Removed redundant CLIP encodings for improved efficiency
- Images encoded exactly once, features reused throughout
- Enhanced documentation with sklearn-based calibration guidance
- Added comprehensive test suite

## 📚 Citation

If you use this package in your research, please cite:

```bibtex
@article{megahed2025conformal,
  title={Conformal Prediction for Vision-Based Few-Shot Defect Classification},
  author={Megahed, Fadel M. and Chen, Ying-Ju and Yousif, Ibrahim and
          Colosimo, Bianca Maria and Grasso, Marco Luigi Giuseppe and
          Jones-Farmer, L. Allison},
  journal={arXiv preprint arXiv:2501.12596},
  year={2025},
  url={https://arxiv.org/abs/2501.12596}
}
```

## 👥 Authors & Maintainers

- **Fadel M. Megahed** (fmegahed@miamioh.edu)
- **Ying-Ju (Tessa) Chen** (ychen4@udayton.edu)
- **Ibrahim Yousif**

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🔗 Links

- **PyPI**: https://pypi.org/project/conformal-clip/
- **GitHub**: https://github.com/fmegahed/conformal_clip
- **Documentation**: https://fmegahed.github.io/research/conformal_clip/
- **Paper**: https://arxiv.org/abs/2501.12596

---

**Full Changelog**: https://github.com/fmegahed/conformal_clip/compare/320e83b...v0.1.1
