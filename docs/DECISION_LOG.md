# DECISION LOG
**Project:** FPGA Transformer Fault Diagnosis
**Maintainer:** Mehrdad Sheikh Abbasi

This file records confirmed design and implementation decisions.
Decisions are added when the user explicitly confirms or approves something.

Format per entry:
- **Date** — when the decision was made or confirmed
- **Decision** — what was decided
- **Why** — reason / motivation
- **Source** — where the decision came from (proposal, code, user confirmation, etc.)
- **Affects** — which files or stages are impacted

---

## Dataset and Classification

### D001 — Four-class grouped classification
- **Date:** (project start)
- **Decision:** Use 4 grouped classes: normal, imbalance, misalignment, bearing_fault
- **Why:** Project proposal specifies coarse fault type detection, not severity subclasses
- **Source:** `MehrdadSheikhAbbasi_Proposal_v3_final.pdf`, `src/config.py::LABEL_MAP`
- **Affects:** All stages (data, model output, FPGA constants, PC display)

### D002 — All 8 MAFAULDA channels as model input
- **Date:** (project start)
- **Decision:** Use all 8 channels (tachometer + 6 accel + microphone), `CHANNEL_SELECTION = "all"`
- **Why:** Multi-sensor input; `vibration_only` mode not yet implemented
- **Source:** `src/config.py`, CLAUDE.md
- **Affects:** `prepare_dataset.py`, `make_dataloaders.py`, Transformer input_dim

### D003 — File-level train/val/test split (70/10/20)
- **Date:** (project start)
- **Decision:** Split at file level (not window level) to prevent data leakage
- **Why:** Windows from the same file are highly correlated; window-level split would leak
- **Source:** `src/data/make_dataloaders.py::split_file_index`
- **Affects:** All data modules, evaluation fairness

---

## Preprocessing

### D004 — Window size 1024, stride 1024 (non-overlapping)
- **Date:** (project start, active as of Day 11)
- **Decision:** `WINDOW_SIZE = 1024`, `WINDOW_STRIDE = 1024`
- **Why:** Non-overlapping windows avoid redundancy; 1024 is compatible with Transformer; model reaches ~98% with this size
- **Source:** `src/config.py`
- **Affects:** All data modules, Transformer seq_len, FPGA buffer design (future)
- **Revisit flag:** Window size is an architectural decision that interacts with Transformer sequence length (1024 tokens → large attention matrix). After learning more about Transformer internals, revisit whether 1024 is optimal or whether a different size (e.g., 512) would be better for FPGA latency/resource tradeoff. Changing this invalidates the current cache and requires full retraining.

### D005 — Per-file z-score normalization
- **Date:** (active as of Day 11)
- **Decision:** Normalize each CSV file independently before windowing
- **Why:** Each file has its own operating conditions; per-file normalization keeps relative signal structure
- **Source:** `src/data/prepare_dataset.py::normalize_dataframe`, `artifacts/results/preprocessing_config.json`
- **Note:** This deviates from the CLAUDE.md Stage 2 spec (which says fit on train, apply to val/test). Needs user confirmation.
- **Affects:** `prepare_dataset.py`, `preprocessed_cache.py`, FPGA normalization path (future)

### D006 — Signal processing flags all disabled (no resample, no filter, no Wiener)
- **Date:** (active as of Day 11)
- **Decision:** All optional preprocessing flags in `src/config.py` are `False`
- **Why:** Frequency analysis was done (spectrum_report.json) but no resampling decision was finalized
- **Source:** `src/config.py`, `artifacts/results/preprocessing_config.json`
- **Affects:** `preprocessed_cache.py`, cache invalidation if flags change

### D011 — Optional backshift tail window (default OFF)
- **Date:** 2026-06-15
- **Decision:** Add configurable `INCLUDE_TAIL_WINDOW` (default `False`) and `TAIL_WINDOW_MODE = "backshift"`. When enabled, after the normal full windows, if trailing real samples remain (file length not divisible by the windowing schedule), one extra full-size window aligned to the file end (`start = num_rows - window_size`) is added. No zero-padding; every window stays `[window_size, num_channels]`. The single guard `tail_start > last_standard_start` ensures a tail is added only when a non-zero remainder exists and the tail is not a duplicate of the last standard window.
- **Why:** At the current 1024/1024 config each 250 000-row MAFAULDA file discards 144 trailing samples (~0.058%) and the tail adds only +1 window/file (244→245, +0.41%), so no measurable accuracy change is expected. The option is added for **generality**: the discarded fraction grows with larger `window_size` (planned 1024 vs 2048 study, D004 revisit) or larger stride, where silently dropping the tail becomes wasteful. Backshift is preferred over zero-/reflect-padding because it uses only real samples and does not distort the signal/spectrum. Default is OFF to preserve the existing windowing behavior (CLAUDE.md: do not silently change windowing).
- **Safety:** File-level split is unchanged; the tail window only overlaps other windows **within the same file**, so it never crosses train/val/test (no leakage). Counts are recorded in `build_dataloaders` stats (`tail_window_info`) and in `preprocessing_config.json`.
- **Source:** `src/config.py`, `src/data/prepare_dataset.py::create_windows_from_dataframe`, `src/data/make_dataloaders.py`, `src/data/pipeline_validation.py`
- **Tests:** `python -m src.tools.test_tail_window` (all cases pass: short/equal/divisible/non-divisible/overlap/unsupported-mode).
- **Ablation:** `python -m src.tools.ablation_tail_window` compares tail off vs on (train/val/test accuracy, macro/weighted F1, confusion matrices). **Status: NOT yet executed** — the model ablation needs the training environment (torch + GPU; the full run takes hours). Data-level effect is exact and verified on real files: every file → 244 normal + 1 tail window, so for the current split (train 1365 / val 195 / test 391 files) the enabled run adds train 333 060→334 425, val 47 580→47 775, test 95 404→95 795 windows (+0.41% each).
- **Affects:** Windowing, dataloader stats, augmentation cache key, future FPGA buffer/windowing logic.

---

## Model Architecture

### D007 — Lightweight Transformer Encoder
- **Date:** Day 4
- **Decision:** `d_model=32`, `nhead=4`, `num_layers=2`, `dim_feedforward=64`, `dropout=0.1`
- **Why:** Small enough for FPGA feasibility; achieves ~98% accuracy on MAFAULDA
- **Source:** `src/config.py`, `src/models/transformer_baseline.py`
- **Affects:** Quantization (Stage 6), FPGA resource estimate (Stage 8)

### D008 — Mean pooling over sequence for classification
- **Date:** Day 4
- **Decision:** Use `x.mean(dim=1)` over the time axis before the classifier head
- **Why:** Simple, hardware-friendly, avoids [CLS] token complexity
- **Source:** `src/models/transformer_baseline.py`
- **Affects:** FPGA accumulator design (future)

---

## Training

### D009 — Adam optimizer, lr=1e-3, 6 epochs
- **Date:** (active as of Day 11)
- **Decision:** `LEARNING_RATE = 1e-3`, `NUM_EPOCHS = 6`
- **Why:** Config comment says "temporary for fast testing" — needs user confirmation if this is final
- **Source:** `src/config.py`, `src/train_transformer_baseline.py`
- **Note:** Best epoch was 6 of 6 — training may have been cut short
- **Affects:** `transformer_baseline_best.pt` validity

### D010 — Class-weighted CrossEntropyLoss when imbalance ratio > 1.5
- **Date:** (Stage 4)
- **Decision:** Auto-apply inverse-frequency weights when max/min class ratio > 1.5
- **Why:** Severe imbalance (22:1, bearing_fault vs normal); unweighted loss biases toward majority class
- **Source:** `src/train_transformer_baseline.py::build_loss_function`
- **Affects:** Training; imbalance_info saved in metrics

### D012 — Experiment config snapshot saved with every training/evaluation run
- **Date:** 2026-06-15
- **Decision:** The training script embeds `model_config`, `training_config`, and `preprocessing_config` (plus `optimizer_state_dict`) inside `transformer_baseline_best.pt`, and also writes standalone `artifacts/checkpoints/transformer_baseline_model_config.json` and `artifacts/results/transformer_experiment_config.json`. All values are read from `src/config.py` via `src/experiment_config.py` — never hard-coded — so the snapshot always matches the actual run. Evaluation reads the embedded `model_config` to rebuild the architecture, warning and falling back to `config.py` if absent.
- **Why:** The previous checkpoint stored only weights; if `config.py` changes later, the architecture that produced a checkpoint becomes ambiguous. Snapshotting the config makes runs reproducible and supports Stage 6/7 (quantization, golden vectors).
- **Backward compatibility:** Old checkpoints without `model_config` still load (eval prints a warning and uses `config.py`). The existing `transformer_baseline_best.pt` was NOT modified; its standalone JSON was regenerated from `config.py` (architecture matches the known baseline). Retraining is only needed to embed the snapshot inside that `.pt`.
- **Source:** `src/experiment_config.py`, `src/train_transformer_baseline.py`, `src/evaluate/evaluate_model.py`
- **Affects:** Reproducibility; downstream quantization/FPGA constant extraction.

---

## Confirmed by User — 2026-06-14

### C001 — Training is NOT finalized
- **Decision:** Current `transformer_baseline_best.pt` is not the final model. Training config (epochs, batch size, architecture) needs review.
- **Why:** Training time is very long; hyperparameters are not optimized; user wants to find better settings before final run.
- **Affects:** Stages 5 (reproducibility), 6 (quantization), and everything downstream.

### C002 — Hyperparameter tuning not completed
- **Decision:** No final hyperparameters decided yet. `tune_transformer_baseline.py` exists but tuning is in progress.
- **Why:** Right hyperparameters haven't been found yet.
- **Affects:** `src/config.py` (TRANSFORMER_*, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)

### C003 — Log txt files moved to notes/run_logs/
- **Decision:** The 6 root-level `.txt` terminal capture files moved to `notes/run_logs/` with a README index.
- **Why:** They were cluttering the root; kept for reference.
- **Affects:** File layout only; no code impact. `tune_baselin2e.txt` renamed to `tune_baseline2.txt`.

---

## Hardware / FPGA Deployment — 2026-08-16

### D013 — FPGA-deployed model is F_1layer (not the 2-layer baseline)
- **Date:** 2026-08-16
- **Decision:** The hardware/HLS core targets `F_1layer` (1 encoder layer,
  d_model=32, 4 heads, ffn=64, 9,492 params), not the higher-accuracy 2-layer
  baseline (98.22%).
- **Why:** F halves the encoder hardware to fit the small EBAZ4205 / Zynq-7010
  (80 DSP, ~270 KB BRAM); full-test accuracy stays high (float 97.55%).
- **Source:** User confirmation (2026-08-16); `kaggle/` sweeps.
- **Affects:** Stages 6-8, `hls_f_int10/`, `pynq/`. The day14 A_baseline int16
  core in `hls/` is preserved untouched.

### D014 — Numeric format is int10
- **Date:** 2026-08-16
- **Decision:** Quantize activations, weights and attention to **int10**
  (`ap_int<10>`).
- **Why:** Bit-width sweep `[16,14,12,10,8]` showed int10 is the smallest
  near-lossless width (full-test **97.11%**, drop 0.44%, no class collapse);
  int8 collapses the imbalance class (recall 0.66). HLS supports arbitrary
  widths, so int10 is free in hardware and lighter than int16.
- **Source:** `kaggle/sweep_multibit/`, `kaggle/int10_shard/int10_fulltest_combined.json`.
- **Affects:** `hls_f_int10/` export + core, `hls_f_int10/vitis/ap_types.h`.

### D015 — Full-test int10 measured by 4-way Kaggle CPU sharding
- **Date:** 2026-08-16
- **Decision:** The definitive int10 full-test number is combined from 4 sharded
  CPU kernels, not a single run or a subset.
- **Why:** The int64 NumPy reference is ~0.42 s/window; all 95,404 windows exceed
  Kaggle's 12 h per-session wall, but its weekly quota is GPU-only so CPU runs
  are free. Strided shards keep class balance; raw counts combine exactly.
- **Source:** User request (2026-08-16); `kaggle/int10_shard/`.
- **Affects:** Reported accuracy; `docs/final_report.md`.

---

## Pending Decisions (awaiting user input)

| ID | Question | Related observation |
|----|----------|---------------------|
| P001 | Is per-file z-score normalization an accepted decision, or should we revisit it? | Explained below; needs decision before FPGA path |
| P003 | Should old data paths (`data/processed/`, `mafaulda_synthetic/`) be kept or removed? | Not used by active pipeline |
| P005 | Should `docs/` be committed to git? | Currently entirely untracked |
| P006 | Is it intentional that `CLAUDE.md` is in `.gitignore`? | Not tracked by git |
| P008 | Should the 3 uncommitted source changes be committed? | `src/config.py` staged; 2 others unstaged |

---

## Normalization Question (P001) — Explanation

**What the code currently does:**
Each CSV file is normalized on its own. For example, if file A has mean=5 and std=2, it gets normalized using those numbers. If file B has mean=100 and std=50, it gets normalized using those different numbers. Each file is treated independently.

**What CLAUDE.md Stage 2 says should happen:**
Compute mean and std only from training files. Then use those same fixed numbers to normalize validation and test files too.

**Why it matters for FPGA:**
- Current (per-file): FPGA receives raw sensor data → must first read all 1024 samples, compute mean/std, then normalize → more logic, extra buffering step.
- Alternative (fixed global stats): FPGA has hardcoded constants (e.g., mean=X, std=Y per channel, computed once from training) → subtract and divide by fixed numbers → much simpler hardware.

**Practical impact on software accuracy:** Probably small, because each file already comes from a controlled test condition with consistent amplitude. But the decision affects what the FPGA needs to implement.

**No decision needed right now** — training isn't finalized yet. This can be decided when we prepare for Stage 6 (quantization). Flagged here so it isn't forgotten.
