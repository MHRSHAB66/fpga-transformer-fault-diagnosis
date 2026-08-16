# Day 9 Notes

## Main focus
Day 9 was a second cleaning-and-improving pass, focused on the Stage 2
preprocessing pipeline and making data loading fast and reproducible.

## What was done
- Implemented lazy loading for windowed datasets
- Added caching of preprocessed files and training augmentations
- Added multiprocessing for preprocessing the dataset
- Added an optional signal-processing pipeline plus validation tools for Stage 2
- Enhanced analysis and training scripts (representative plots, detailed evals)
- Updated the model training configuration and data-loading process

## Current result
Preprocessing became much faster (caching + multiprocessing + lazy loading) and
the Stage 2 pipeline gained validation tooling, which made repeated experiments
practical.

## Final note
Day 9 was mostly about throughput and reproducibility of the data path -- the
groundwork that made the later frequency study and tuning affordable.
