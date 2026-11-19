# FPGA Transformer Fault Diagnosis

Real-time machinery condition monitoring using a lightweight Transformer
classifier deployed on FPGA hardware, using the MAFAULDA vibration dataset.

**B.Sc. project — Mehrdad Sheikh Abbasi, Amirkabir University of Technology.**

## Project goal

A complete engineering pipeline from data to a demonstrable system:

```
MAFAULDA windows -> preprocessing/windowing -> lightweight Transformer (PyTorch)
  -> evaluation on unseen files -> quantization -> FPGA inference
  -> result sent to PC -> simple PC-side display
```

The implemented system runs an INT10 Transformer inference core on a physical
MYIR Z-Turn XC7Z020 board and presents its predictions on a computer.
**Baseline deployed-system results, with evidence:**
[`docs/reports/consolidated_results_20261022.md`](docs/reports/consolidated_results_20261022.md).
Demo operation: [`docs/demo_runbook.md`](docs/demo_runbook.md).
Later raw-input, temporal-vote, noise-robustness, and endurance studies:
[`docs/reports/paper_readiness_review_20261110.md`](docs/reports/paper_readiness_review_20261110.md).

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
| 2 | done | Preprocessing, windowing, DataLoaders |
| 3–4 | done | MLP sanity baseline + lightweight Transformer baseline |
| 5 | done | Reproducibility, config snapshots, final software evaluation |
| 6–8 | done | INT10 quantization, golden vectors, HLS core; physically run on the board, class decided in FPGA logic (`pred_class` register) since 2026-10-23 |
| 9 | partly | Preprocessing inside the FPGA from raw samples (board-validated); physical sensor optional |
| 10–11 | done | Results to PC over the USB-UART and over SSH/TCP Ethernet; Replay/Live viewer |
| 12 | done (prepared windows) | Integration and validation; see the consolidated results for limits |

Key results of the **baseline** deployed system (MYIR Z-Turn XC7Z020;
see the consolidated results for test conditions and evidence):

| Metric | Value |
|---|---|
| Accuracy / macro-F1 / macro AUC | 97.77 % / 95.65 % / 0.9991 |
| Board output vs Python INT10 reference | identical on all windows, incl. all 4 logits |
| Sustained throughput (board) | 60.9 windows/s vs 48.83 required (50 kHz, 1,024 stride) |
| Hard deadline, board side (RAM input, SCHED_FIFO) | 0 misses in 100,000 windows, max 16.56 ms of 20.48 ms |
| FPGA resources (post-route, 100 MHz) | LUT 59.7 %, FF 29.1 %, BRAM 50.4 %, DSP 95.5 % |

Later, separately validated variants and additional experiments (not the same
measurement as the baseline rows above):

| Experiment | Result | Evidence |
|---|---|---|
| Raw-input core (on-FPGA z-score + INT10) | 97.7705 % on 95,404 board windows; all logits match C-model; post-route LUT 68.9 %, BRAM 54.6 % | `docs/reports/onchip_preprocessing_20261109.md` |
| Temporal vote, deployed-model weights, continuous stream, k=5 | 99.084 % *system* accuracy (per-window model: 97.7705 %) | `docs/reports/paper_readiness_review_20261110.md` section 6 |
| Noise-augmented variant with its own FPGA bitstream, 0-dB input, vote k=5 | 93.835 % system accuracy (per-window: 90.1304 %) | `docs/reports/paper_readiness_review_20261110.md` section 5 |
| Board-boundary endurance, RAM input, SCHED_FIFO | 0 deadline misses in 1,400,000 windows; worst 19.25 ms of 20.48 ms | `docs/reports/paper_readiness_review_20261110.md` section 7 |
| Same model on the board's ARM CPU | ~219 ms/window (FPGA ~14x faster in the cited comparison) | `docs/reports/cpu_baseline_20261109.md` |

Figures: `docs/figures/final/` (regenerate with `python -m tools.plot_final_figures`).

Per-stage detail: `docs/project_status.md`. Design decisions: `docs/DECISION_LOG.md`.
[Hardware versions and evidence index](docs/reports/hardware_evidence_index.md) distinguishes
HLS estimates, Vivado post-route results, and physical board measurements.
Project stage plan and engineering constraints: `AGENTS.md`.

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

**Run automated tests (no model training):**
```bash
python -m unittest discover -s tests
```
These tests need the project's Python dependencies; some hardware-dependent tests
use mocks. This does **not** replace testing on the physical board. Historical
one-off preprocessing scripts, including `test_tail_window.py`, are under
`archive/`. The saved preprocessing config is
`artifacts/results/preprocessing_config.json`.

**Train the Transformer baseline:**
```bash
python -m src.train_transformer_baseline
```
Training creates a timestamped `artifacts/runs/<run_name>/` directory, prints
its path, and stores the best checkpoint alongside the model, training, and
preprocessing configuration snapshots.

**Evaluate that checkpoint:**
```bash
python -m src.evaluate.evaluate_model --checkpoint "artifacts/runs/REPLACE_WITH_RUN_DIR/checkpoints/transformer_best.pt"
```
Replace `REPLACE_WITH_RUN_DIR` with the actual directory printed by training.
The evaluator writes a fresh timestamped evaluation under `artifacts/runs/` and
uses the checkpoint's recorded `model_config` when available. The evaluator's
older default checkpoint path remains available for backward compatibility,
but is not where new training runs are saved.

## Artifacts

A fresh run is organized as follows (the actual run directory is timestamped):

```
artifacts/runs/<run_name>/
  checkpoints/transformer_best.pt       # model weights and config snapshot
  checkpoints/transformer_baseline_model_config.json  # architecture snapshot
  results/metrics.json                  # training + evaluation metrics
  results/transformer_experiment_config.json         # experiment configuration
  results/history.json                  # training history
  plots/                                # training and confusion-matrix figures
  logs/                                 # runtime summaries
```

Historical experiment outputs, FPGA build reports, hardware deployments and
board tests use other directories under `artifacts/`. See the
[hardware evidence index](docs/reports/hardware_evidence_index.md) for the
hardware-report locations and source labels.
