# Day 4 Notes

## Main focus
Day 4 focused on adding a simple Transformer-based baseline to the project and verifying that it works with the existing data pipeline.

## What was done
- Created a new Transformer baseline model
- Added a simple positional encoding module
- Projected the input features into a higher-dimensional embedding space
- Used a Transformer encoder for sequence modeling
- Applied mean pooling over the time dimension
- Added a classifier head for 4-class prediction
- Created a separate training script for the Transformer baseline
- Reused the existing training pipeline, checkpoint saving, history saving, and detailed evaluation setup
- Successfully trained and evaluated the Transformer baseline on the synthetic dataset

## Current result
The Transformer model ran successfully, saved outputs correctly, and integrated with the project without runtime issues. However, validation and test performance were very weak, which is expected because the current dataset is still tiny and synthetic.

## Artifacts
- `artifacts/checkpoints/transformer_baseline_best.pt`
- `artifacts/results/transformer_baseline_history.json`

## Final note
Day 4 was mainly about architectural integration and pipeline verification for a Transformer-based model, not about achieving meaningful performance yet.