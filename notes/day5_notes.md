# Day 5 Notes

## Main focus
Day 5 moved the project off the tiny synthetic dataset and onto the real
MAFAULDA main dataset.

## What was done
- Switched the pipeline to work on the main MAFAULDA dataset
- Updated dataset directory references to use `DATA_RAW_DIR` (single config source)
- Synced the classification setup with the real dataset
- Verified the existing loading/windowing path runs on real files

## Current result
The pipeline now reads and processes the real MAFAULDA CSV files instead of
synthetic data, which is the prerequisite for any meaningful accuracy.

## Final note
Day 5 was a data-source switch: from a placeholder synthetic set to the real
dataset the rest of the project is built on.
