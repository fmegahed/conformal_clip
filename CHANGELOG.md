# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2025-11-16

### Added
- Backends module (`conformal_clip/backends.py`) to load CLIP-like models via open-clip-torch and vision-only models via timm.
  - Exposes `load_backend`, `VISION_LANGUAGE_BACKENDS`, and `VISION_ONLY_BACKENDS`.
  - Supports openai, openclipbase, siglip2, eva-clip, mobileclip2, dinov3, mobilenetv4, plus custom ids.
  - Loads HF token from `.env` via `HF_TOKEN` for gated models.
- Benchmark utility (`conformal_clip/benchmark.py`) to compare backends across calibration (None/isotonic/sigmoid) and conformal modes (None/global/mondrian) on fixed splits.
  - Returns classification and conformal metrics DataFrames plus styled views with yellow-highlighted best values.
- Examples in `examples/`:
  - `textile_openai.py`, `textile_openclipbase.py`, `textile_siglip2.py`, `textile_eva_clip.py`, `textile_mobileclip2.py`, `textile_dinov3.py`, `textile_mobilenetv4.py`.
  - `benchmark_textile.py` demonstrating the full benchmark and saving HTML with highlights.

### Changed
- `wrappers.encode_and_normalize` now adds a batch dimension for 3D tensors before `encode_image`.
- `zero_shot.evaluate_zero_shot_predictions` accepts a `tokenize_fn` compatible with OpenCLIP; still supports `clip_module.tokenize` if provided.
- Public API now exports `load_backend`, `VISION_LANGUAGE_BACKENDS`, `VISION_ONLY_BACKENDS`, and `benchmark_models`.
- README modernized to clarify OpenCLIP and timm usage and to document the new benchmark.

### [Unreleased]

### Added
- New small backends for low-resource environments in `conformal_clip/backends.py`:
  - CLIP-like: `clip_b32` (`ViT-B-32-quickgelu`), `clip_b16` (`ViT-B-16-quickgelu`).
  - Vision-only (timm): `resnet18` (`resnet18.a1_in1k`), `efficientnet_b0` (`efficientnet_b0.ra_in1k`).
- Resource-tiered benchmarking in `conformal_clip/benchmark.py` via `resource_tier` argument:
  - `"low"`: default; small CLIP/timm models only (e.g., `clip_b32`, `clip_b16`, `siglip2`, `mobileclip2`, `mobilenetv4`, `resnet18`, `efficientnet_b0`).
  - `"medium"`: low-tier plus mid-size models that should run on ~8–12 GB GPUs (e.g., `openai`, `resnet50`, `coca`, `dinov3`).
  - `"high"`: all non-custom backends including heavy models (e.g., `openclipbase`, `vitg`, `eva02`, `convnext`).

### Changed
- `benchmark_models` now defaults `resource_tier` to `"low"` when `backends` is not provided, to avoid loading very large models by default.
- Updated README benchmark example and resource warning to explain `resource_tier` and its default.



## [0.1.1] - 2025-11-06

### Added

#### Core Functionality
- Initial public release to PyPI
- Few-shot CLIP classification using vision encoder only (no text encoding for test images)
- Conformal prediction with both Global and Mondrian (class-conditional) modes
- Probability calibration via isotonic regression or sigmoid (Platt) scaling
- Comprehensive metrics computation for point predictions and conformal sets
- Zero-shot evaluation baseline for comparison
- Support for custom datasets via `load_image` utility
- Optional textile defect dataset via `conformal-clip[data]` installation

#### Documentation
- Complete README rewrite with step-by-step usage guide (5 detailed steps)
- Beta release notice with warning badge prominently displayed
- sklearn-based calibration guidance citing sklearn documentation
- Notes on 100-sample calibration experiment results (isotonic vs sigmoid)
- BibTeX citation section for research use
- Enhanced docstrings for all public functions in conformal.py
- Improved CLIPWrapper class documentation in wrappers.py
- Added comprehensive docstrings to utility functions in image_io.py, viz.py, metrics.py
- CONTRIBUTING.md with development setup, code style, and submission guidelines

#### Examples & Testing
- `examples/` directory with 3 demonstration scripts:
  - `basic_usage.py`: Minimal 50-image example
  - `textile_inspection.py`: Full workflow comparing calibration methods
  - `custom_dataset.py`: Template for user's own data
- `examples/README.md` with examples documentation
- `tests/` directory with comprehensive pytest suite (15 tests, 30% coverage):
  - `test_imports.py`: Import smoke tests
  - `test_wrappers.py`: CLIPWrapper functionality validation
  - `test_metrics.py`: Utility function tests
  - `test_image_io.py`: Image loading tests
- `pytest.ini` configuration file

#### GitHub Actions & CI/CD
- `.github/workflows/publish-to-pypi.yml`: Automated publishing on releases
- `.github/workflows/publish-to-test-pypi.yml`: Manual test publishing workflow
- `.github/workflows/tests.yml`: Automated testing on Python 3.9-3.12
- `.github/RELEASE_NOTES_v0.1.1.md`: Ready-to-use release notes
- `.github/RELEASE_TEMPLATE.md`: Template for future releases
- `.github/GITHUB_ACTIONS_SETUP.md`: Complete setup instructions

#### Package Metadata
- `__all__` exports in __init__.py for explicit public API
- MANIFEST.in for controlling distribution contents
- Comprehensive .gitignore (expanded from 7 to 163 lines covering Python, IDEs, OS files)
- Enhanced pyproject.toml with:
  - URLs (Homepage, Repository, Documentation, Bug Tracker)
  - Classifiers for Python 3.9-3.12
  - CLIP as git dependency: `clip @ git+https://github.com/openai/CLIP.git`
  - Ibrahim Yousif added as author
- Package exclusions to prevent development files in distributions

#### Integration
- Integration with conformal-clip-data v0.1.4
- Updated API usage: `nominal_dir()`, `local_dir()`, `global_dir()` instead of `get_textile_base_dir()`
- Reproducible sampling workflow using `sample_urls` with `np.random.default_rng(2024)`
- Explicit results directory creation: `os.makedirs("results", exist_ok=True)`

### Fixed
- Removed redundant CLIP encodings for improved efficiency
- Images are now encoded exactly once and features are reused throughout
- Removed __pycache__/ bytecode files from git tracking
- Excluded development files from distributions (index.qmd, index.html, index_files/)
- Fixed date inconsistencies in release documentation

### Changed
- Enhanced docstrings for all public functions with detailed examples
- Improved README with clear focus on manufacturing inspection and occupational safety applications
- Added CLIP dependencies (ftfy, regex, tqdm) to package requirements
- Updated from internal `get_textile_base_dir()` to external `conformal_clip_data` API
- Updated all examples and README to use new data package imports
- Changed workflow from basic examples to comprehensive reproducible sampling approach

## [0.1.0] - 2025-11-03

### Added
- Initial implementation with core functionality
- CLIPWrapper sklearn-compatible classifier
- Conformal prediction utilities
- Metrics computation functions
- Visualization helpers