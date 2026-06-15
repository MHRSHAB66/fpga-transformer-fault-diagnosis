# FPGA Transformer Fault Diagnosis

Real-time machinery condition monitoring using a lightweight Transformer
classifier, prepared for FPGA execution, on the MAFAULDA vibration dataset.

**B.Sc. project — Mehrdad Sheikh Abbasi, Amirkabir University of Technology.**

## Project goal

A complete engineering pipeline from data to a demonstrable system:

```
MAFAULDA windows -> preprocessing/windowing -> lightweight Transformer (PyTorch)
  -> evaluation on unseen files -> quantization -> FPGA inference
  -> result sent to PC -> simple PC-side display
```

The final target is a Transformer-based fault classifier prepared for FPGA, with
its predictions shown on a computer. This repository currently covers the
**software phase** (data → training → evaluation → reproducibility). The
quantization, FPGA, communication, and UI stages are future work.

## Dataset

[MAFAULDA](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html) machinery fault
database. Each file is a multichannel time-series (8 columns: tachometer, 6
accelerometers, microphone) sampled at 50 kHz. Raw data is **not** stored in the
repository (it is git-ignored under `data/`); download it separately.

Files are grouped into 4 classes: `normal`, `imbalance`, `misalignment`,
`bearing_fault`. The train/validation/test split is performed at **file level**
(not window level) to prevent leakage between splits.

## Pipeline stages

| Stage | Status | Description |
|------|--------|-------------|
| 0–1 | done | Project setup, data discovery, class mapping |
| 2 | in progress | Preprocessing, windowing, DataLoaders (signal-processing experiments optional) |
| 3–4 | done | MLP sanity baseline + lightweight Transformer baseline |
| 5 | in progress | Reproducibility, config snapshots, final software evaluation |
| 6–12 | TODO | Quantization, FPGA inference, FPGA↔PC communication, PC UI |

See `docs/DECISION_LOG.md` for confirmed design decisions and `CLAUDE.md` for the
full stage plan and constraints.

## Installation

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/mac: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then set PROJECT_ROOT to this directory's absolute path
```

`requirements.txt` lists only the core runtime dependencies. `librosa` is
optional and used solely by the experimental sound/MFCC feature-extraction
scripts; install it manually only if you run those.

## Usage

All commands run from the project root.

**Data validation (no training):**
```bash
python -m src.tools.test_pipeline          # build a small subset, check shapes, one forward pass
python -m src.data.pipeline_validation     # export artifacts/results/preprocessing_config.json
python -m src.tools.test_tail_window       # windowing unit checks (numpy only)
```

**Train the Transformer baseline:**
```bash
python -m src.train_transformer_baseline
```
Saves the best checkpoint plus an embedded reproducibility snapshot
(`model_config`, `training_config`, `preprocessing_config`).

**Evaluate a saved checkpoint:**
```bash
python -m src.evaluate.evaluate_model
```
Loads `artifacts/checkpoints/transformer_baseline_best.pt`, reports accuracy,
precision/recall/F1 and confusion matrices. Uses the checkpoint's embedded
`model_config` when present, otherwise falls back to `src/config.py`.

## Artifacts

```
artifacts/checkpoints/transformer_baseline_best.pt           # best model weights + config snapshot
artifacts/checkpoints/transformer_baseline_model_config.json # standalone architecture config
artifacts/results/transformer_baseline_metrics.json          # training metrics
artifacts/results/transformer_experiment_config.json         # full experiment snapshot
artifacts/results/transformer_evaluation.json                # evaluation metrics
artifacts/results/*.png                                       # loss/accuracy/confusion-matrix plots
artifacts/cache/                                              # per-file preprocessing cache (git-ignored)
```

Configuration constants live in `src/config.py`. Reproducibility snapshots are
built from it by `src/experiment_config.py`, so saved configs always match the
run.
