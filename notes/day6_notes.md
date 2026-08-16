# Day 6 Notes

## Main focus
Day 6 added the advanced preprocessing and training-support machinery
(signal processing, augmentation, class-imbalance handling).

## What was done
- Added signal preprocessing functions (filtering / signal-processing utilities)
- Added feature-extraction and preprocessing modules for the signal data
- Added data-augmentation functions (noise injection and SMOTE) for the training split
- Implemented class-weight computation and a class-weighted loss for imbalance
- Updated the data-splitting logic and the model-saving mechanism (better
  validation tracking)
- Enhanced the model architecture alongside the preprocessing changes

## Current result
The preprocessing and training path became configurable and imbalance-aware.
Several of these (SMOTE, some signal-processing options) were later treated as
optional experiments rather than the default pipeline.

## Final note
Day 6 was the "advanced preprocessing" build-out; later cleanup days pruned the
parts that did not earn their place in the main FPGA-bound pipeline.
