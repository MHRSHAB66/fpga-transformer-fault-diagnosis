# Day 11 Notes

## Main focus
Day 11 finalized the software/model phase: a reproducible, presentation-ready
training run plus hyperparameter tuning.

## What was done
- Made the software phase reproducible and presentation-ready (deterministic
  seeds, saved config beside checkpoint, clean metrics)
- Finalized the software training artifacts and cleaned generated files
- Added an FPGA-friendly hyperparameter tuning kernel (grid of ~6 configs)
- Added tuning results and summaries for the tuning run

## Current result
The final software model was locked in: a lightweight Transformer classifier at
**98.22%** test accuracy on 95,404 unseen windows (see
`docs/final_software_result.md`), with reproducible artifacts under
`artifacts/runs/`.

## Final note
Day 11 marked the end of the pure-software phase; from here the project turns to
quantization and the FPGA path. (This is the last state `main` sat at for a
while, by design.)
