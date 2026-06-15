# Run Logs

Terminal output captures from various pipeline runs.
These are for reference only — no code depends on them.
All runs below were on the D:\ path (old machine location, before project moved to G:\).

---

| File | Script | Conditions | Key numbers |
|------|--------|-----------|-------------|
| `dataloader_cold.txt` | `python -m src.data.make_dataloaders` | Cold run — no cache, 1951 files, 4 workers | build: 1877s, first batch: 19.9s |
| `dataloader_warm.txt` | `python -m src.data.make_dataloaders` | Warm run — cache already built | build time much faster |
| `train_baseline.txt`  | MLP baseline training | First run on real MAFAULDA | — |
| `train_baseline2.txt` | MLP baseline training | Second run | — |
| `tune_baseline.txt`   | Transformer hyperparameter tuning | First tuning run | — |
| `tune_baseline2.txt`  | Transformer hyperparameter tuning | Second tuning run | original filename had a typo: `tune_baselin2e.txt` |

Note: Files contain garbled characters in some editors because they were captured
with Windows UTF-16 encoding.
