# Day 3 Notes

## Main focus
Day 3 focused on improving the baseline training pipeline and making it more reproducible and easier to analyze.

## What was done
- Reviewed the project structure and current pipeline again
- Cleaned up path printing in inspection utilities
- Added proper seed setup for Python, NumPy, PyTorch, and CUDA
- Added best-model tracking based on validation accuracy
- Saved the best model checkpoint during training
- Reloaded the best model before final test evaluation
- Added training history saving in JSON format
- Added confusion matrix and per-class accuracy for test evaluation

## Current result
The baseline MLP still overfits the tiny synthetic dataset, which is expected. Training accuracy becomes very high, while validation and test performance remain weak. At this stage, the baseline is mainly useful for verifying that the pipeline works correctly.

## Artifacts
- `artifacts/checkpoints/mlp_baseline_best.pt`
- `artifacts/results/mlp_baseline_history.json`

## Final note
Day 3 was mainly about making the baseline pipeline cleaner, more stable, and more interpretable before moving to more advanced models.