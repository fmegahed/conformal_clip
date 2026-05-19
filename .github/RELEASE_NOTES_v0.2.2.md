# Release v0.2.2 — Optional `requests` Fix and README Restructure

## Overview

This is a patch release focused on a packaging bug fix and a documentation overhaul. It is fully backward compatible with `0.2.1` — no public API changed.

## Fixed

- **`requests` is now correctly optional.** `conformal_clip/image_io.py` and `conformal_clip/io_github.py` previously imported `requests` at module top, but `requests` is not declared in package dependencies. On Python 3.10+ this raised `ModuleNotFoundError: No module named 'requests'` at import time, breaking calls to `load_image()` and `get_image_urls()` and failing CI on those Python versions. Both modules now lazy-import `requests` only when a URL is actually fetched, matching the optional-HTTP-stack design described in `__init__.py`. If a URL is passed without `requests` installed, a clear `ImportError` is raised pointing the user at `pip install requests`.

## Changed

### README restructure

The README has been reorganized so that every code block is runnable in the order it appears:

- **`Setup` now precedes `Quickstart`.** Every variable used in a code block is defined before it appears.
- **`Setup` is a numbered flow:** Step 1 (device + backend), Step 2A *or* 2B (example dataset or your own folders — both runnable, both produce the same path lists), and Step 3 (PIL banks, labels, filenames).
- **PIL vs. tensor callout** clarifies that `benchmark_models` takes PIL images while `few_shot_fault_classification_conformal` and `evaluate_zero_shot_predictions` take preprocessed tensors.
- **Each Quickstart block** is labeled "Continues from Setup above" and applies `preprocess_fn` only where the API requires tensors.
- **`Examples`** expanded to cover the textile, extrusion, microstructure, and pipe datasets, grouped by purpose (per-backend few-shot, benchmark suites, discovery/customization).
- **Custom-models section** reworked: the previously heading-less timm block now has its own subsection with a `custom-vision` load example and a note that vision-only backends cannot be used for zero-shot.
- **`Project structure`** updated to include `image_io.py`, `io_github.py`, `data_utils.py`.
- **Documentation badge** added linking to the docs site already listed in `pyproject.toml`.

### Changelog cleanup

- Folded the orphaned `### [Unreleased]` block (small backends `clip_b32`/`clip_b16`/`resnet18`/`efficientnet_b0` and the `resource_tier` argument) into the `0.2.0` entry, where git history confirms those items actually shipped.
- Normalized date dashes across older entries to match the em-dash style used in `0.2.1` and `0.2.2`.

## Notes

- All changes are backward compatible.
- Users who rely on `load_image(url)` or `get_image_urls(...)` must have `requests` available (it was always a runtime requirement for URL paths; this release just makes the failure mode explicit rather than silently breaking at import).
