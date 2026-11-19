# PROJECT AUDIT

> **Historical audit (2026-06-14), not current project status.** This document preserves an early review and its then-open tasks. For current status use [`project_status.md`](project_status.md), and for post-board measurements use [`reports/consolidated_results_20261022.md`](reports/consolidated_results_20261022.md) and [`reports/paper_readiness_review_20261110.md`](reports/paper_readiness_review_20261110.md).
**Date:** 2026-06-14
**Auditor:** Claude (Sonnet 4.6)
**Branch at audit time:** `day11-code-review-and-presentation-prep`
**Status:** Phase 1 complete — awaiting user confirmation before Phase 2

---

## Project Goal

Real-time machinery condition monitoring using a lightweight Transformer trained on MAFAULDA 8-channel vibration data. Final goal: FPGA inference + UART output + PC display.

Four-class classifier:
- `0` — normal
- `1` — imbalance
- `2` — misalignment
- `3` — bearing_fault

---

## Directory Tree (clean)

```
fpga_fault_diagnosis/
├── .env                            ← PROJECT_ROOT path (gitignored)
├── .gitignore
├── CLAUDE.md                       ← project instructions (gitignored!)
├── LICENSE
├── README.md                       ← 2-line stub, minimal
├── requirements.txt                ← 7 deps: numpy, pandas, sklearn, matplotlib, torch, scipy, librosa
│
├── artifacts/
│   ├── cache/                      ← gitignored (large; do not hand-edit)
│   │   ├── preprocessed_files/     ← active per-file numpy cache (~1951 .npz files)
│   │   └── train_augmentation/     ← augmented training cache (disabled by default)
│   ├── checkpoints/
│   │   ├── mlp_baseline_best.pt
│   │   ├── transformer_baseline_best.pt          ← main trained model
│   │   └── transformer_baseline_model_config.json  ← NEW, untracked in git
│   └── results/
│       ├── preprocessing_config.json             ← pipeline config snapshot
│       ├── mlp_baseline_history.json
│       ├── transformer_baseline_history.json
│       ├── transformer_baseline_metrics.json     ← training metrics
│       ├── transformer_evaluation.json           ← NEW, untracked in git
│       ├── transformer_baseline_*.png            ← loss/accuracy/confusion plots
│       ├── dataloader_*.png
│       ├── channel_analysis/                     ← Stage 2: discriminativity analysis
│       │   ├── discriminativity_scores.json
│       │   ├── channel_analysis_summary.json
│       │   ├── column_identity_verification.json
│       │   └── per_channel_psd/                  ← 8 PSD plots per class
│       ├── frequency_investigation/              ← Stage 2: FFT + axial investigation
│       └── spectrum_analysis/                    ← Stage 2: full spectrum report
│           └── spectrum_report.json              ← resampling recommendations
│
├── data/
│   ├── raw/
│   │   ├── mafaulda/               ← REAL dataset (gitignored), 1951 CSV files, 4 fault classes
│   │   └── mafaulda_synthetic/     ← OLD synthetic debug data (4 folders of CSVs)
│   ├── processed/                  ← OLD experimental preprocessing output (not active pipeline)
│   │   ├── feature_extraction_summary.json
│   │   ├── features_dataset.npz
│   │   ├── preprocessing_manifest.json
│   │   ├── preprocessed_signals/   ← individual sample .npz files (old path)
│   │   └── .augmented_train_cache/ ← OLD cache location (superseded by artifacts/cache/)
│   └── sample/                     ← mini synthetic dataset + generator script (debug)
│       ├── generate_mini_dataset.py
│       ├── manifest.csv
│       └── {bearing_fault,imbalance,misalignment,normal}/
│
├── docs/                           ← ENTIRELY UNTRACKED in git (newly added)
│   ├── project_definition/
│   │   ├── extracted/
│   │   │   ├── Project Definition.md
│   │   │   └── proposal_full.md
│   │   └── raw/
│   │       ├── MehrdadSheikhAbbasi_Proposal_v3_final.pdf  ← primary source of truth
│   │       └── Project Definition.pdf
│   └── references/
│       ├── extracted/
│       │   ├── Fault Detection in Industrial Machinery.md
│       │   └── INTELLIGENT FAULT DIAGNOSIS OF TYPE AND SEVERITY IN.md
│       └── raw/
│           ├── Fault Detection in Industrial Machinery.pdf
│           └── INTELLIGENT FAULT DIAGNOSIS OF TYPE AND SEVERITY IN.pdf
│   [MISSING] project_definition/notes/  ← mentioned in CLAUDE.md but folder does not exist
│   [MISSING] references/notes/          ← mentioned in CLAUDE.md but folder does not exist
│
├── experiments/                    ← empty
├── notebooks/                      ← empty
│
├── notes/
│   ├── day1_notes.md
│   ├── day2_notes.md
│   ├── day3_notes.md
│   └── day4_notes.md               ← last note; days 5–11 are undocumented
│
├── src/
│   ├── config.py                   ← CENTRAL CONFIG (all constants, flags, paths)
│   ├── train_mlp_baseline.py       ← MLP training script
│   ├── train_transformer_baseline.py  ← main Transformer training + per-epoch eval
│   ├── tune_transformer_baseline.py   ← hyperparameter tuning script
│   ├── data/
│   │   ├── __init__.py
│   │   ├── augmentation.py         ← SMOTE / noise injection (optional, disabled)
│   │   ├── cached_augmentation.py  ← augmentation cache management
│   │   ├── load_data.py            ← CSV discovery + label inference
│   │   ├── make_dataloaders.py     ← pipeline orchestrator (main entry point)
│   │   ├── pipeline_validation.py  ← validation utilities
│   │   ├── prepare_dataset.py      ← preprocessing helpers + windowing
│   │   ├── preprocessed_cache.py   ← per-file numpy disk cache (active)
│   │   └── windowed_dataset.py     ← lazy-loading dataset (alternative path)
│   ├── evaluate/
│   │   ├── __init__.py
│   │   └── evaluate_model.py       ← standalone eval: accuracy + P/R/F1 + confusion matrix
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp_baseline.py
│   │   └── transformer_baseline.py ← main model (d_model=32, 2 layers, 4 heads)
│   └── tools/                      ← one-off Stage 2 analysis scripts
│       ├── analyze_mafaulda_spectrum.py
│       ├── channel_analysis.py
│       ├── check_data_quality.py
│       ├── extract_features.py     ← possibly experimental/old
│       ├── frequency_investigation.py
│       ├── inspect_data.py
│       ├── preprocess_data.py      ← possibly experimental/old
│       ├── signal_preprocessing.py ← possibly experimental/old
│       └── test_pipeline.py
│
├── dataloader_cold.txt  ← captured terminal output (D:\ paths = OLD machine location)
├── dataloader_warm.txt  ← captured terminal output (D:\ paths = OLD machine location)
├── train_baseline.txt   ← captured terminal output
├── train_baseline2.txt  ← captured terminal output
├── tune_baseline.txt    ← captured terminal output
└── tune_baselin2e.txt   ← captured terminal output [typo in filename]
```

---

## Current Pipeline (Active Path)

```
MAFAULDA CSVs (data/raw/mafaulda/)
    ↓
load_data.py         → build file index + infer labels (normal/imbalance/misalignment/bearing_fault)
    ↓
make_dataloaders.py  → file-level stratified split (70/10/20)
    ↓
preprocessed_cache.py → per-file: load CSV → numeric → drop if >20% missing → fill NaN →
                         z-score normalize → optional signal processing (currently all disabled)
                         → cache as .npz in artifacts/cache/preprocessed_files/
    ↓
prepare_dataset.py   → windowing: 1024-sample non-overlapping windows (stride=1024)
                        shape: [N, 1024, 8] — 8 channels, float32
    ↓
augmentation.py      → optional SMOTE or noise injection (train split only, disabled by default)
    ↓
TensorDataset → DataLoader  (batch_size=8, shuffle=True for train only)
    ↓
transformer_baseline.py  → Linear(8→32) + PositionalEncoding + 2×TransformerEncoderBlock(d=32, h=4, ff=64) + mean pooling + Linear head
    ↓
train_transformer_baseline.py  → Adam(lr=1e-3), CrossEntropyLoss with class weights, 6 epochs
                                  saves best checkpoint by val_loss
    ↓
evaluate_model.py    → load checkpoint → compute accuracy + precision + recall + F1 (macro/weighted) + confusion matrix
    ↓
artifacts/checkpoints/transformer_baseline_best.pt
artifacts/results/transformer_baseline_metrics.json
artifacts/results/transformer_evaluation.json
```

---

## Stage Status (vs CLAUDE.md)

| Stage | CLAUDE.md claim | Reality |
|-------|----------------|---------|
| 0 — Setup | COMPLETE | ✅ Confirmed. Config, dirs, .env, requirements all present. |
| 1 — Data discovery | COMPLETE | ✅ Confirmed. 1951 files discovered, 4-class label inference working. |
| 2 — Preprocessing/Windowing/DataLoader | IN PROGRESS | ⚠️ Partially confirmed (see notes below). |
| 3 — MLP baseline | COMPLETE | ✅ Confirmed. Checkpoint + history exist. |
| 4 — Transformer baseline | COMPLETE | ✅ Confirmed. Test acc ≈ 98.21%, checkpoint + metrics exist. |
| 5 — Reproducibility + final eval | IN PROGRESS / VERIFY | ⚠️ Partially done (see notes below). |
| 6 — Quantization | TODO | ❌ No code found. |
| 7 — Golden vectors / HW spec | TODO | ❌ No code found. |
| 8 — FPGA inference core | TODO | ❌ No code found. |
| 9 — Sensor input | TODO | ❌ No code found. |
| 10 — FPGA-to-PC UART | TODO | ❌ No code found. |
| 11 — PC-side UI | TODO | ❌ No code found. |
| 12 — Final integration | TODO | ❌ No code found. |

### Stage 2 Details
- ✅ CSV loading, windowing (1024-sample, non-overlapping), DataLoader: working
- ✅ File-level split (no leakage): confirmed
- ✅ Spectrum analysis (FFT, PSD per channel per class): done, report saved
- ✅ Channel discriminativity study (8 channels ranked): done
- ✅ Preprocessing config exported to JSON
- ⚠️ Normalization is **per-file z-score** (computed on each file independently), NOT fitted on train split and applied to val/test. This deviates from the Stage 2 requirement in CLAUDE.md and is noted in `preprocessing_config.json`.
- ⚠️ Resampling analysis done (spectrum_report.json has recommendations) but **resampling not yet applied** (`RESAMPLE_ENABLED = False`)
- ⚠️ Lowpass filter, Wiener filter, low-energy filter are implemented and configurable but **all disabled** (`*_ENABLED = False`)
- ❓ Window length / stride comparison (1024 vs 2048) not yet documented as a completed experiment

### Stage 5 Details
- ✅ Deterministic seed: implemented
- ✅ Confusion matrix: generated for val and test
- ✅ Model config saved beside checkpoint
- ✅ Evaluation script (`evaluate_model.py`) produces F1, precision, recall
- ⚠️ `transformer_evaluation.json` is new and **untracked** in git (likely just generated)
- ⚠️ `NUM_EPOCHS = 6` in config has a comment "temporary for fast testing" — unclear if this is the final value
- ⚠️ `BATCH_SIZE = 8` is very small for 333K training samples

---

## Key Model Numbers (from transformer_baseline_metrics.json)

| Metric | Value |
|--------|-------|
| Val accuracy | 98.21% |
| Test accuracy | 98.21% |
| Best epoch | 6 of 6 |
| Class imbalance ratio | 22:1 (normal vs bearing_fault) |
| Training samples | 333,060 windows |
| Test samples | 95,404 windows |
| Window shape | [1024, 8] |
| Training time | ~10,720 seconds (~3 hours) |

---

## Active Code Files (do not modify without audit)

- [src/config.py](../src/config.py) — all constants live here
- [src/data/make_dataloaders.py](../src/data/make_dataloaders.py) — pipeline orchestrator
- [src/data/prepare_dataset.py](../src/data/prepare_dataset.py) — preprocessing + windowing
- [src/data/preprocessed_cache.py](../src/data/preprocessed_cache.py) — per-file cache
- [src/data/load_data.py](../src/data/load_data.py) — file index + label inference
- [src/models/transformer_baseline.py](../src/models/transformer_baseline.py) — model
- [src/train_transformer_baseline.py](../src/train_transformer_baseline.py) — training
- [src/evaluate/evaluate_model.py](../src/evaluate/evaluate_model.py) — evaluation

---

## Generated Artifacts (do not hand-edit)

- `artifacts/checkpoints/transformer_baseline_best.pt`
- `artifacts/checkpoints/transformer_baseline_model_config.json`
- `artifacts/results/transformer_baseline_metrics.json`
- `artifacts/results/transformer_baseline_history.json`
- `artifacts/results/transformer_evaluation.json`
- `artifacts/results/preprocessing_config.json`
- `artifacts/results/spectrum_analysis/spectrum_report.json`
- `artifacts/results/channel_analysis/discriminativity_scores.json`
- All `*.png` files under `artifacts/results/`
- All `.npz` files under `artifacts/cache/preprocessed_files/`

---

## Possibly Extra or Outdated Files

| File / Folder | Status | Reason |
|---------------|--------|--------|
| `data/raw/mafaulda_synthetic/` | Likely obsolete | Old debug data; current pipeline uses real MAFAULDA only |
| `data/processed/` | Likely obsolete | Old experimental preprocessing path; active cache is in `artifacts/cache/` |
| `data/sample/` | Debug only | Mini synthetic dataset for early dev; not used in main pipeline |
| `dataloader_cold.txt` | Informal log | Terminal output capture; references old D:\ path |
| `dataloader_warm.txt` | Informal log | Same as above |
| `train_baseline.txt` | Informal log | Terminal output |
| `train_baseline2.txt` | Informal log | Terminal output |
| `tune_baseline.txt` | Informal log | Terminal output |
| `tune_baselin2e.txt` | Informal log (typo) | Terminal output; filename has typo |
| `experiments/` | Empty | No content |
| `notebooks/` | Empty | No content |
| `notes/day1_notes.md` – `day4_notes.md` | Archived | Only days 1–4 documented; days 5–11 missing |
| `src/tools/extract_features.py` | Unclear | May be experimental only |
| `src/tools/preprocess_data.py` | Unclear | May be experimental only |
| `src/tools/signal_preprocessing.py` | Unclear | May be experimental only |

---

## Known Observations and Risks

1. **CLAUDE.md is gitignored** — project instructions are not committed to the repository. If lost, they cannot be recovered from git history.

2. **docs/ folder is entirely untracked** — all PDFs and extracted markdowns are outside git. If this is intentional (PDFs are large), it should be documented.

3. **Per-file normalization** — normalization statistics are computed per CSV file, not fitted on the training split. This is documented in `preprocessing_config.json` but deviates from Stage 2 spec. No obvious data leakage for classification, but worth noting for hardware path (FPGA needs to know what normalization parameters to use).

4. **Project was moved from D:\ to G:\** — `dataloader_cold.txt` and `dataloader_warm.txt` contain `D:\` paths; `.env` has `G:\`. All currently-generated artifacts use `G:\`. The old logs are just informational and do not affect the pipeline.

5. **NUM_EPOCHS = 6 is labeled "temporary for fast testing"** — unclear whether this was the final training run or if more epochs are planned.

6. **BATCH_SIZE = 8 is very small** — with 333K training samples, this means 41,633 batches per epoch. May be a GPU-memory constraint on this machine.

7. **`tune_transformer_baseline.py` exists but its role is unclear** — not audited in Phase 1; needs review in Phase 3.

8. **`windowed_dataset.py` exists as an alternative lazy dataset** — the active pipeline uses materialized `TensorDataset`. This lazy alternative may have been an experiment.

9. **Three uncommitted changes at session start**:
   - `src/config.py` — staged
   - `src/evaluate/evaluate_model.py` — unstaged (working tree)
   - `src/train_transformer_baseline.py` — unstaged (working tree)

---

## Source Documents (docs/ folder)

| File | Role |
|------|------|
| `docs/project_definition/raw/MehrdadSheikhAbbasi_Proposal_v3_final.pdf` | **Primary source of truth** for project requirements |
| `docs/project_definition/raw/Project Definition.pdf` | Supporting definition; proposal overrides if conflict |
| `docs/project_definition/extracted/proposal_full.md` | Markdown extraction of proposal |
| `docs/project_definition/extracted/Project Definition.md` | Markdown extraction of project definition |
| `docs/references/raw/Fault Detection in Industrial Machinery.pdf` | Background reference (Koutoulas thesis) |
| `docs/references/raw/INTELLIGENT FAULT DIAGNOSIS OF TYPE AND SEVERITY IN.pdf` | Background reference (Spadini paper) |
| `docs/references/extracted/*.md` | Markdown extractions of reference papers |

**Missing:** `docs/project_definition/notes/` and `docs/references/notes/` folders mentioned in CLAUDE.md do not exist.

---

## Open Questions for User (Phase 1 Confirmation)

See also: [DECISION_LOG.md](DECISION_LOG.md)

1. **txt log files at root** — `dataloader_cold.txt`, `dataloader_warm.txt`, `train_baseline.txt`, `train_baseline2.txt`, `tune_baseline.txt`, `tune_baselin2e.txt`: are these still needed or can they be cleaned up?

2. **Old data paths** — `data/raw/mafaulda_synthetic/` and `data/processed/`: are these still needed, or can they be considered obsolete?

3. **Per-file normalization** — the active pipeline normalizes each CSV file independently (per-file z-score), not globally on the training split. Is this an accepted decision, or do you want to revisit it?

4. **NUM_EPOCHS = 6 "temporary"** — was the current `transformer_baseline_best.pt` trained with the "final" config, or is a new training run planned with more epochs?

5. **Uncommitted changes** — `src/config.py` (staged), `src/evaluate/evaluate_model.py` and `src/train_transformer_baseline.py` (unstaged): do you want to commit these, or is there a specific reason they are uncommitted?

6. **docs/ untracked in git** — is it intentional that all project documentation (PDFs, markdowns) is outside git? Should it be committed?

7. **CLAUDE.md gitignored** — is it intentional that project instructions are not tracked by git?

8. **tune_transformer_baseline.py** — is this still actively used? Is the hyperparameter tuning complete or in progress?

---

## Suggested Next Steps (after Phase 1 confirmation)

- Phase 2: Review data / preprocessing / windowing / DataLoader in detail
- Phase 3: Review models / training scripts
- Phase 4: Review evaluation / artifacts / current metrics
- Phase 5: Plan for Stage 6 (quantization) onwards
- Cleanup: Decide on obsolete files, commit pending changes, improve README
