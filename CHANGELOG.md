# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
