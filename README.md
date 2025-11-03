# conformal_clip

Utilities for zero-shot and few-shot CLIP classification with optional **conformal prediction**, **probability calibration**, **metrics**, and **visualization**.

---

## 🚀 Features
- **Zero-shot evaluation** with CLIP text prompts.  
- **Few-shot classification** using CLIP image exemplars.  
- **Conformal prediction** (Global and Mondrian) for set-valued predictions with finite-sample guarantees.  
- **Optional probability calibration** (isotonic or sigmoid) before conformal scoring.  
- **Comprehensive metrics** for both point predictions and conformal sets.  
- **Simple I/O utilities** for local or URL-based image access and GitHub folder listings.  
- **Optional visualizations**, including confusion matrices and coverage summaries.

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
results/                            # Example experiment outputs (CSV + plots)
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
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install CLIP:
   ```bash
   pip install git+https://github.com/openai/CLIP.git
   ```

---

## 🧠 Quickstart

### **Zero-Shot Evaluation**
```python
import torch, clip
from conformal_clip import evaluate_zero_shot_predictions

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

labels = ["Nominal", "Defective"]
test_images = [preprocess(PIL_image).unsqueeze(0).to(device) for PIL_image in images]
test_filenames = [...]
label_counts = [n_nominal, n_defective]

metrics_df, results_df = evaluate_zero_shot_predictions(
    labels, label_counts, test_images, test_filenames, model, device, clip_module=clip,
    save_confusion_matrix=True, cm_file_path="results", cm_file_name="confusion_matrix.png"
)
```

---

### **Few-Shot + Conformal Prediction**
```python
from conformal_clip import few_shot_fault_classification_conformal

results = few_shot_fault_classification_conformal(
    model=model,
    test_images=test_images,
    test_image_filenames=test_filenames,
    nominal_images=nominal_bank,              # list of preprocessed tensors
    nominal_descriptions=["...", "..."],      # optional text prompts
    defective_images=defective_bank,
    defective_descriptions=["...", "..."],
    calib_images=calib_images,                # list of calibration tensors
    calib_labels=calib_labels,                # list of "Nominal"/"Defective"
    alpha=0.1,
    temperature=1.0,
    mondrian=True,                            # class-conditional thresholds
    prob_calibration="isotonic",              # None | "isotonic" | "sigmoid"
    allow_empty=False,
    csv_path="results",
    csv_filename="exp_results_conformal.csv",
    print_one_liner=True
)
```

The resulting CSV includes one row per test image:
```
datetime_of_operation, alpha, temperature, mondrian,
image_path, image_name, point_prediction, prediction_set,
set_size, Nominal_prob, Defective_prob,
nominal_description, defective_description
```

---

## 📊 Metrics

### **Point Prediction Metrics**
```python
from conformal_clip import compute_classification_metrics

m = compute_classification_metrics(
    csv_file="results/exp_results_conformal.csv",
    labels=["Nominal", "Defective"],
    label_counts=[n_nominal, n_defective],
    save_confusion_matrix=True,
    cm_file_path="results",
    cm_file_name="exp_conf_matrix.png",
    cm_title="Confusion Matrix for Experiment"
)
```

### **Conformal Set Metrics**
```python
from conformal_clip import compute_conformal_set_metrics

cm = compute_conformal_set_metrics(
    csv_file="results/exp_results_conformal.csv",
    labels=["Nominal", "Defective"],
    label_counts=[n_nominal, n_defective]
)
```

Outputs include accuracy, recall, specificity, precision, F1-score, AUC (binary), and coverage (overall and per class).

---

## 🧩 Notes & Tips

- **Preprocessing:** Always use CLIP’s provided `preprocess` before passing images to `encode_image`.  
- **Temperature:** The `temperature` parameter controls softmax sharpness.  
- **Probability Calibration:**  
  - `"isotonic"` — best for larger calibration sets, preserves monotonicity.  
  - `"sigmoid"` — stable for smaller calibration sets, uses Platt-style scaling.  
- **Mondrian vs Global:**  
  - *Mondrian* = per-class coverage.  
  - *Global* = overall coverage.  
- **Allow Empty:**  
  - `False` → always outputs a label (forces argmax).  
  - `True` → allows abstention (empty conformal set).  

---

## 📘 Quarto Demo

The included **`index.qmd`** demonstrates:
- Loading CLIP and preprocessing data  
- Running few-shot + conformal experiments  
- Computing metrics and visualizing results  

Rendered output: `index.html`

---

## 📚 Citation

If you use this code, please cite the authors listed in `index.qmd`:

> Fadel M. Megahed, Ying-Ju (Tessa) Chen, Bianca Maria Colosimo,  
> Marco Luigi Giuseppe Grasso, and L. Allison Jones-Farmer.

---

## ⚖️ License
This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.
