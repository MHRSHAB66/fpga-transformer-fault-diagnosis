# Day 2

## Completed
- Added full data quality checking across all CSV files
- Added preprocessing into the data pipeline:
  - numeric conversion
  - missing-value handling
  - file dropping based on missing ratio threshold
  - normalization
- Implemented file-level train/validation/test split
- Built PyTorch DataLoaders
- Implemented and tested a baseline MLP classifier
- Verified that the end-to-end training pipeline runs successfully on the synthetic dataset

## Current pipeline
raw CSV files  
→ quality check  
→ preprocessing  
→ windowing  
→ file-level split  
→ DataLoader  
→ baseline model training

## Current preprocessing choices
- Missing fill method: median
- Missing ratio threshold: 0.2
- Normalization: z-score
- Window size: 128
- Window stride: 128

## Current dataset setup
- Using synthetic dataset: `data/raw/mafaulda_synthetic`
- 4 classes:
  - normal
  - imbalance
  - misalignment
  - bearing_fault
- Windowed input shape: `(128, 8)`

## Baseline training result
- Device: CUDA
- Train samples: 20
- Validation samples: 4
- Test samples: 8

Final baseline result:
- train accuracy reached 1.00
- validation accuracy stayed low
- test accuracy = 0.25

## Interpretation
- The training pipeline works correctly
- The baseline model can overfit the tiny synthetic dataset
- Validation/test performance is not meaningful yet because the dataset is very small and synthetic
- This stage was for pipeline verification, not real performance evaluation

## Important notes for later
- Before finalizing preprocessing choices on the real dataset, review the relevant papers
- Filtering is not needed yet
- Feature extraction is not needed yet
- Keep reminding to check the papers before locking:
  - normalization method
  - window size / stride
  - overlap usage
  - channel selection
  - split strategy
  - class balancing strategy