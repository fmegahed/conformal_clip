# Release v0.2.1 - Benchmark Output Controls and Traceability Improvements

## Overview

This release focuses on improving control over benchmark output, enhancing filename stability, and increasing traceability in conformal predictions. The changes help streamline experiment tracking and make repeated benchmarking runs easier to organize.

## Highlights

### Benchmark Output Controls

- Added `csv_path` argument to `benchmark_models`.  
  Setting `csv_path=None` disables all CSV output.  
  When saving is enabled, the output directory is created automatically.

- Added `csv_prefix` argument for safely customizing output filenames.  
  Users can prepend identifiers such as `run1_`, `expA_`, or `baseline_`  
  without changing the underlying filename structure.

- Retained a stable output naming pattern via `csv_filename_template`.  
  Filenames follow the pattern `{backend}_{cal}_{conf}.csv`,  
  and `csv_prefix` is prepended when provided.

### Conformal Output Enhancements

- Improved exemplar traceability by replacing empty exemplar descriptions with  
  auto-generated labels:
  - `Nominal: nominal_i`  
  - `Defective: defective_i`  
  This applies to all conformal output and improves interpretability of results.

### Documentation

- Updated the `benchmark_models` docstring to describe `csv_path`, `csv_prefix`,  
  the filename template, and multi-CSV output behavior.

## Notes

- All changes are backward compatible.  
- Benchmark naming consistency is preserved across versions while remaining customizable.  
- These updates are especially helpful for automated experiments, multi-run evaluations,  
  and environments where output directories must be controlled precisely.
