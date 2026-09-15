# Final Software/Model Phase Audit

**Date:** 2026-06-16
**Branch:** `day11-code-review-and-presentation-prep`
**Purpose:** Decision document before the final Kaggle T4 training run. Audits the active pipeline, separates real code from experimental code, evaluates paper/docs ideas, and proposes the **minimum** set of changes/ablations needed to honestly call the software/model phase complete.
**Scope rule:** No training was run. No code was changed. This is inspection only.

> Read alongside [PROJECT_AUDIT.md](PROJECT_AUDIT.md) (structure) and [DECISION_LOG.md](DECISION_LOG.md) (confirmed decisions D001–D012, C001–C003, P001).

---

## 1. Short diagnosis of the current pipeline

The main pipeline is **clean, single-path, and learnable**. File-level split, per-file z-score normalization, fixed 1024-sample non-overlapping windows, all-8-channel input, a genuinely small Transformer (≈ tens of thousands of params), weighted cross-entropy for the 22:1 imbalance, best-by-val-loss checkpointing, and a full metrics/confusion/F1 evaluation path. The reproducibility scaffolding (D012: embedded `model_config`/`training_config`/`preprocessing_config`) is in place. The Kaggle smoke test reuses the real code unchanged and passed on T4.

**The phase is NOT yet defensible, for three concrete reasons:**

1. **Training is not converged / not final.** `NUM_EPOCHS = 6` is explicitly labeled *"temporary for fast testing"* and the best epoch was **6 of 6** — i.e. validation loss was still the best on the last epoch, so the model was very likely still improving when training stopped. You cannot claim "final software model" from a run that was cut off at its best epoch with no convergence evidence and no early stopping. (Confirmed not-final by C001.)
2. **The current saved metrics are from the old `D:\` machine**, not from the controlled final Kaggle run. They must be regenerated once, cleanly, and **not overwrite** the historical artifact.
3. **The normalization scheme is not reproducible at FPGA inference time.** Per-file z-score (D005) requires buffering an entire ~250k-row file to compute mean/std. The FPGA sees one 1024-sample window at a time. If you keep per-file normalization, the software result and the hardware result are computed under *different* normalization and are not directly comparable — and fixing it later forces a full retrain. This is the one preprocessing decision that should be settled **before** the final run, not deferred.

Everything else (architecture, split, imbalance handling, augmentation-off, filters-off) is defensible as-is. The dominant risk is **#1 (epochs/convergence)**; the dominant *hidden* risk is **#3 (normalization vs hardware)** plus a **Kaggle RAM risk** (Section 12).

---

## 2. Files/functions ACTIVE in the main pipeline

Verified by import graph from `train_transformer_baseline.py` and `evaluate/evaluate_model.py`.

| File | Active functions | Role |
|------|------------------|------|
| `src/config.py` | all constants | Central config; every constant lives here |
| `src/data/load_data.py` | `build_file_index`, `infer_label_from_path`, `load_one_csv`, `project_relative_or_absolute`, `list_csv_files` | CSV discovery + file-level label inference |
| `src/data/prepare_dataset.py` | `preprocess_dataframe`, `convert_dataframe_to_numeric`, `should_drop_file`, `fill_missing_values`, `normalize_dataframe`, `create_windows_from_dataframe`, `count_normal_windows` | Per-file preprocessing + windowing |
| `src/data/preprocessed_cache.py` | `load_or_create_preprocessed_array` (+ key/load/save helpers) | Per-file `.npz` disk cache (active) |
| `src/data/make_dataloaders.py` | `build_dataloaders`, `build_X_y_from_file_index`, `split_file_index`, `safe_train_test_split`, `deterministic_small_split` | **Pipeline orchestrator (main entry)** |
| `src/models/transformer_baseline.py` | `TransformerBaseline`, `TransformerEncoderBlock`, `PositionalEncoding` | **Main model** |
| `src/train_transformer_baseline.py` | `train`, `run_training`, `build_loss_function`, `compute_class_weights`, `evaluate`, `evaluate_with_details`, `set_seed`, plot helpers | **Main training + per-epoch eval** |
| `src/experiment_config.py` | `build_model_config`, `build_training_config`, `build_preprocessing_config`, `save_experiment_config_files` | Reproducibility snapshot (D012) |
| `src/evaluate/evaluate_model.py` | `evaluate_saved_model`, `_confusion_to_f1`, `_build_model_from_config` | Standalone final eval (P/R/F1/confusion) |

**Conditionally active (wired in, disabled by config flags — they ARE part of the active code path but currently do nothing):**

- `src/data/augmentation.py` (`augment_training_data`, `apply_smote`, `apply_noise_injection`) and `src/data/cached_augmentation.py` — only invoked when `USE_SMOTE_AUGMENTATION` or `USE_NOISE_INJECTION` is True. Both are `False`.
- `prepare_dataset.py`: `apply_lowpass_filter`, `apply_wiener_filter`, `apply_resample_array`, `apply_low_energy_filter` — invoked inside `load_or_create_preprocessed_array` only when the corresponding `*_ENABLED` flag is True. All `False`.

---

## 3. Files/functions EXPERIMENTAL or INACTIVE (not in the main path)

| File | Status | Evidence |
|------|--------|----------|
| `src/data/windowed_dataset.py` (`WindowedFileDataset`) | **Inactive / dead in main path** | Not imported by any active module. `make_dataloaders` always materializes a `TensorDataset`. Referenced only in docstrings + `pipeline_validation.py`/tools. *(Note: this is the lazy/streaming alternative — relevant to the RAM risk in §12.)* |
| `src/data/pipeline_validation.py` | Test-only | Imported only by `src/tools/test_pipeline.py` |
| `src/tune_transformer_baseline.py` | Experimental | Standalone tuner; not part of final path. Tuning declared incomplete (C002) |
| `src/models/mlp_baseline.py`, `src/train_mlp_baseline.py` | Stage 3 sanity baseline | Not the final model; keep as baseline evidence |
| `src/tools/analyze_mafaulda_spectrum.py`, `channel_analysis.py`, `frequency_investigation.py` | One-off Stage 2 analysis | Produced the spectrum/discriminativity reports; not in training path |
| `src/tools/extract_features.py`, `preprocess_data.py`, `signal_preprocessing.py` | **Old feature-engineering path** | Write to `data/processed/`; superseded by the cache pipeline. Pull in `librosa`/MFCC-style ideas — NOT the project target |
| `src/tools/inspect_data.py`, `check_data_quality.py`, `test_pipeline.py`, `test_tail_window.py`, `ablation_tail_window.py` | Dev/QA utilities | Manual tools; `ablation_tail_window` never executed (D011) |
| `data/processed/`, `data/raw/mafaulda_synthetic/`, `data/sample/` | Obsolete/debug data | Not used by active pipeline (P003) |

**Targeted answers to your "specifically audit" list:**

- **Augmentation code** (`augmentation.py`): correct, train-only, off by default. SMOTE flattens each `[1024,8]` window to an 8192-D vector and linearly interpolates raw waveforms between neighbors — this **destroys phase/spectral structure and creates physically unrealistic vibration signals**. Not needed (see §6).
- **Cached augmentation** (`cached_augmentation.py`): correct, keyed on train file list + preprocessing config; only touched when augmentation is on. Inert.
- **Signal preprocessing/filtering** (`apply_lowpass/wiener/resample/low_energy`): implemented, cache-key-aware, all disabled. `filtfilt`/Wiener are **non-causal** (explicitly noted in code) → not directly FPGA-reproducible. Treat as software-only ablations.
- **Feature extraction** (`tools/extract_features.py`, `signal_preprocessing.py`): old experimental path, not connected. Do **not** promote (would turn the project into the Spadini feature pipeline — forbidden by CLAUDE.md).
- **Normalization** (`normalize_dataframe`): per-file z-score, computed independently per CSV (D005). Leakage-free for software; not FPGA-reproducible (see §6, §11).
- **Multi-worker / DataLoader** (`DATALOADER_NUM_WORKERS = 0`): train/val/test are **materialized in-RAM TensorDatasets**, so multi-worker gives ~no benefit; `0` is the most reproducible. See §6 and §12.
- **Timing/logging**: `build_dataloaders_seconds`, `first_train_step_seconds`, `val_seconds_history`, `total_training_seconds` are logged to `transformer_baseline_metrics.json`. **Gap:** no per-epoch *train* time (only total + per-epoch val). Minor.
- **Evaluation/metrics** (`evaluate_model.py`): accuracy, per-class P/R/F1, macro + weighted F1, confusion matrices for val and test. Solid. Missing only optional ROC/AUC.
- **Kaggle runner** (`kaggle/kernel/fpga-fault-smoke-test.py`): smoke test only (24 files, 1 epoch). Has a real GPU-compute probe (P100 sm_60 incompatibility handling). Reuses real code unchanged. Does **not** exercise full-dataset memory.
- **Reproducibility/config-saving** (`experiment_config.py`, D012): config embedded in checkpoint + standalone JSON, all read from `config.py`. Good. Seeds set in `set_seed`. **Caveat:** DataLoader `shuffle=True` has no explicit `generator=`; reproducible at `num_workers=0` via global RNG, but would need `worker_init_fn` + a seeded `generator` if workers are ever increased.

---

## 4. Paper/Docs-Based Candidate Improvements

Sources: `docs/references/extracted/Fault Detection in Industrial Machinery.md` (Koutoulas — nine-model comparison incl. Transformer, MAFAULDA + CSU) and `docs/references/extracted/INTELLIGENT FAULT DIAGNOSIS OF TYPE AND SEVERITY IN.md` (Spadini — sound-only MFCC/XGBoost on MAFAULDA microphone). Both are **background only** per CLAUDE.md.

Each candidate evaluated against: (1) used on MAFAULDA/similar vibration? (2) compatible with our time-series Transformer? (3) DL-relevant or feature-ML-only? (4) reduces size/compute/memory? (5) risk of removing useful info (the 22 kHz component)? (6) FPGA-friendly? (7) train/inference/both? (8) integrate / ablation / future / ignore?

### C-1 — Global dataset feature scaling (Min-Max / Robust), train-fitted
- **Source:** Koutoulas §"data rescaling … Min-Max scaler from scikit-learn applied to raw values of each feature"; Robust Scaler used on CSU for outlier robustness. Spadini §4.5 "normalization across the entire dataset."
- **What it does:** One scaler fit on the **training set per channel**, applied to val/test (and reproducible as fixed constants).
- **Already implemented?** Partially — `normalize_dataframe` supports `zscore`/`minmax`, but only **per-file**, never train-fitted/global.
- **Active in main pipeline?** No (per-file z-score is active).
- **Expected benefit:** Directly addresses the CLAUDE.md Stage 2 spec; gives **fixed per-channel constants the FPGA can hardcode** (subtract/divide) → much simpler hardware, and software/hardware become comparable.
- **Risk:** If files have genuinely different operating-point amplitudes, a single global scale may slightly hurt accuracy vs per-file. Requires retraining (cache invalidation).
- **Recommendation:** **Ablation, FPGA-relevant — and a decision to settle before the final run** (see §5/§6). Per-window z-score is the FPGA-faithful alternative.
- **Code area:** `prepare_dataset.normalize_dataframe`, `preprocessed_cache` (fit stats on train split, persist them), `make_dataloaders` (apply train stats to val/test), `experiment_config` (record stats).

### C-2 — Frequency-guided downsampling / anti-alias lowpass
- **Source:** Spadini works down to 8 kHz/8-bit *for the audio channel*; our own `spectrum_report.json` recommends `LOWPASS_CUTOFF_HZ ≈ 22034`, `RESAMPLE_TARGET_SR ≈ 44068` (practical 48000).
- **What it does:** Lowpass then resample the time axis to shrink the sequence.
- **Already implemented?** Yes — `apply_lowpass_filter` + `apply_resample_array`, disabled.
- **Active?** No.
- **Expected benefit:** Shorter sequences → **smaller attention matrix** (big FPGA win, since attention is O(seq²)) and less memory.
- **Risk (critical):** The spectrum report shows `acc_axial_b1` reaches **95 % of its energy at ~22 kHz**, and the microphone is broadband to ~24 kHz. Aggressive downsampling (e.g. the audio-paper's 8 kHz) would **delete the 22 kHz component you specifically observed**. Strict Nyquist for that channel is ~44 kHz, i.e. essentially no downsampling room without loss. `filtfilt` is non-causal → not FPGA-faithful.
- **Recommendation:** **Ablation only, not in the main run.** If tested, the only justified setting is lowpass≈22 kHz + resample≈48 kHz (preserves the band), which gives little size reduction. Reducing **window length** (C-6) is a safer way to shrink the sequence.
- **Code area:** `config.py` flags, `preprocessed_cache` (already wired).

### C-3 — Wiener / wavelet / EMD denoising
- **Source:** Spadini §4.3 (Wiener, grid 1–35, audio); Koutoulas references SVM+EMD, SVM+wavelet (feature-ML methods).
- **Already implemented?** Wiener: yes (`apply_wiener_filter`), disabled. Wavelet/EMD: no.
- **Expected benefit:** Marginal for our case. Koutoulas explicitly argues CNN/Transformer **LayerNorm + learned features already suppress noise**, so explicit denoising is often redundant for DL.
- **Risk:** Per-channel Wiener can remove fault-relevant HF (incl. 22 kHz); non-causal; another retrain.
- **Recommendation:** **Not recommended** for the main run; low-priority ablation at best. Lean on the model's own LayerNorm (the paper's own argument).
- **Code area:** `config.py` flag (already wired).

### C-4 — Oversampling (SMOTE / Borderline-SMOTE / ADASYN / Gaussian-noise balancing)
- **Source:** Spadini §6 (SMOTE family on extracted features); Koutoulas refs [32],[34],[35] (MLP + SMOTE/Gaussian-noise balancing).
- **Already implemented?** SMOTE-like + noise injection: yes, train-only, disabled.
- **Expected benefit:** Low. We already use **inverse-frequency weighted cross-entropy** (D010), and the current run reaches per-class recall 0.956–0.992 despite 22:1 imbalance — imbalance is already handled.
- **Risk:** SMOTE on raw 8192-D waveforms produces **unrealistic vibration signals** (interpolated phase/spectrum). The papers apply SMOTE to *engineered features*, not raw time series — not transferable to our pipeline.
- **Recommendation:** **Not recommended.** Keep augmentation off. Optionally one ablation (noise-injection on) purely to *demonstrate* it doesn't help.
- **Code area:** `config.py` flags (already wired).

### C-5 — Non-overlapping vs overlapping sliding windows
- **Source:** Spadini (NOSW for audio augmentation); Koutoulas ref [36] (time-windowing for representation downsizing).
- **Already implemented?** Yes — `WINDOW_STRIDE` configurable; currently `= WINDOW_SIZE` (non-overlapping). Backshift tail window available, off (D011).
- **Expected benefit:** Overlap multiplies training windows (more data) but **inflates correlation** between windows and is mostly an augmentation trick.
- **Risk:** Overlap within a file is fine (file-level split prevents cross-split leakage), but it biases evaluation if applied to val/test. Adds compute.
- **Recommendation:** **Optional future work.** Non-overlapping is the cleaner, more defensible default. Keep stride = 1024.
- **Code area:** `config.WINDOW_STRIDE`.

### C-6 — Window length study (512 vs 1024 vs 2048)
- **Source:** CLAUDE.md Stage 2 DoD explicitly requires a documented window-length decision; D004 flags revisiting 1024.
- **What it does:** Sets Transformer `seq_len` and the FPGA input buffer.
- **Already implemented?** Configurable (`WINDOW_SIZE`), only 1024 trained.
- **Expected benefit:** **512 halves the sequence → ~4× smaller attention matrix** (1024²→512²), the single biggest FPGA-feasibility lever short of changing the architecture (which is forbidden). 2048 goes the wrong way for hardware.
- **Risk:** Changes accuracy; full retrain + cache rebuild.
- **Recommendation:** **Useful ablation** (512 vs 1024). Picks up the Stage 2 DoD requirement. Do **not** pursue 2048 (worse for FPGA).
- **Code area:** `config.WINDOW_SIZE`/`WINDOW_STRIDE`; model `seq_len` flows automatically.

### Ideas explicitly REJECTED (background-only, would redefine the project)
- Nine-model comparison (Koutoulas), CSU gearbox data, sound-only MFCC/XGBoost pipeline, 8-bit *audio* experiments as the main task (Spadini). All forbidden by CLAUDE.md; ignore for implementation, cite only as literature justification.

---

## 5. Recommended final changes (classified)

### MUST DO before the final run
1. **Set a real epoch budget + early stopping.** Raise `NUM_EPOCHS` (suggest 25–40) and add **early stopping on val loss** with patience (e.g. 5–8) — the infra already tracks best-by-val-loss, so this is a small, safe addition. Without it you cannot claim convergence. *(Requires a small code change — proposed below, not yet applied.)*
2. **Settle the normalization scheme now** (per-file vs per-window vs global train-fitted). It cannot be changed after the final run without retraining, and it determines software↔hardware comparability. See §6 for the recommendation. (P001.)
3. **Version the output artifacts** so the final Kaggle run does **not** overwrite the historical `D:\` metrics. Either write `*_final_t4.json`/checkpoint names or archive the existing `transformer_baseline_*` first.
4. **Use a NEW Kaggle kernel slug** for final training (e.g. `fpga-fault-final-train`), and push the final `src/` as a **new version** of the `fpga-fault-src` dataset. Do **not** overwrite the `fpga-fault-smoke-test` kernel.
5. **Confirm Kaggle RAM headroom** for the materialized tensors (~15.6 GB; §12) before committing a multi-hour run, or switch the train path to the lazy `WindowedFileDataset`.

### USEFUL (defensibility-improving, optional, each = one full run)
6. Normalization ablation (per-window or global-train-fitted vs per-file) — C-1.
7. Window-length ablation 512 vs 1024 — C-6 (also closes the Stage 2 DoD item).

### OPTIONAL / FUTURE
8. Larger `BATCH_SIZE` (e.g. 64–128) to cut wall-clock — the current val step alone is ~93 s/epoch at batch 8. (Changes optimization; treat as part of the final-config choice, not a free change.)
9. Per-epoch train-time logging; optional ROC/AUC in eval.
10. `worker_init_fn` + seeded `generator` *if* `num_workers` is ever raised.

### NOT RECOMMENDED
11. Enabling augmentation (C-4), denoising (C-3), or aggressive downsampling (C-2) in the main run. Architecture changes (forbidden). 2048-length windows.

---

## 6. Direct answers to your "pay special attention to" questions

- **Class imbalance vs augmentation:** Weighted CE already handles the 22:1 imbalance (per-class recall 0.956–0.992). **Augmentation is not needed.** Keep it off.
- **Does augmentation create unrealistic signals?** Yes — SMOTE interpolates raw waveforms (destroys phase/spectrum); noise injection is benign but unhelpful here. Avoid for the main run.
- **Global vs per-window vs per-file normalization:** All three are leakage-safe for software (per-file uses only that file's stats; global must fit on **train only**). The deciding factor is **hardware reproducibility**:
  - *Per-file* (current): FPGA must buffer a whole file → impractical; software≠hardware path.
  - *Global train-fitted*: FPGA hardcodes per-channel mean/std constants → simplest hardware, but assumes amplitude consistency across operating points.
  - *Per-window z-score*: FPGA computes mean/std over the 1024 samples it already buffers → cheap, **exactly reproduces training**, no cross-file assumption.
  - **Recommendation:** strongly consider **per-window z-score** as the final scheme (most FPGA-faithful and reproducible), with **global train-fitted** as the comparison ablation. Decide before the final run.
- **Filtering/downsampling removing the 22 kHz component:** Yes — `acc_axial_b1` holds 95 % energy at ~22 kHz; any cutoff/downsample below ~44 kHz Nyquist deletes it. Keep filters/resampling **off** in the main run; test only as a justified ablation at ≈22 kHz/48 kHz.
- **Multi-worker speed without changing results:** Negligible benefit — datasets are materialized in RAM, so iteration is in-memory; the one-time cost is the single-threaded per-file cache build. Keep `num_workers = 0` (most reproducible). Bigger batch size is the real speed lever.
- **Are all runtime timings logged?** Mostly yes (build, first-step, per-epoch val, total). Missing per-epoch **train** time. Minor.
- **Config suitable for future FPGA inference?** Mostly: small model (d_model 32, 2 layers, 4 heads, ff 64), mean pooling. **Watch-items for later stages (not software changes now):** seq_len=1024 → 1024×1024 attention per head (largest cost; C-6 can halve it); GELU + LayerNorm need hardware approximation; per-file normalization is the main blocker (above).
- **Reproducible & comparable to hardware:** D012 snapshotting is good. To be *comparable to the FPGA*, the **normalization must be reproducible at single-window inference** (favoring per-window). Also fix seeds (done), version artifacts, and freeze the exact checkpoint for Stage 6/7 golden vectors.

---

## 7. Small ablation matrix (≤ 6)

Baseline config A0 is the **proposed final run**; A1–A4 are optional, each a full Kaggle run. Do **not** run all of them — pick at most the two marked ★ if you want maximum defensibility.

| ID | Change vs A0 | Question it answers | Runtime risk | Priority |
|----|--------------|---------------------|--------------|----------|
| **A0** | Final config: epochs 25–40 + early stopping, per-window z-score (or keep per-file — decide §6), batch ≥ 64, filters off, no aug | Is the model converged & defensible? | **Required.** ~1.5–3 h on T4; **RAM ~15.6 GB (verify)** | ★ Required |
| A1 ★ | Normalization = global train-fitted (or per-file) instead of A0's choice | Does normalization scheme change accuracy? Is the FPGA-faithful choice safe? | Med — full retrain, cache rebuild (~2–3 h) | ★ Useful |
| A2 ★ | `WINDOW_SIZE = 512` | Can we 4× shrink attention for FPGA at acceptable accuracy? (Stage 2 DoD) | Med — full retrain, cache rebuild | ★ Useful |
| A3 | Lowpass 22 kHz + resample 48 kHz | Does spectrum-guided anti-alias resampling hurt? | Med–High — cache rebuild + scipy cost; risk of HF loss | Optional |
| A4 | Noise injection ON (train only) | Demonstrate augmentation does **not** help over weighted CE | Med — ~2× train windows → higher RAM/time | Optional |
| A5 | `INCLUDE_TAIL_WINDOW = True` | Does the +0.41% tail data move metrics? (D011) | Low — +0.41% windows only | Optional |

---

## 8. Estimated runtime risk per experiment

- **A0:** Time ~1.5–3 h (lower if batch ≥ 64). **Primary risk = memory**: train tensor ≈ 10.9 GB + val/test ≈ 4.7 GB ≈ **15.6 GB resident** as float32 TensorDatasets, plus CUDA/pin-memory overhead. Verify Kaggle instance RAM (T4 notebooks typically ~29 GB → should fit, tight). Cold per-file cache build of all 1951 CSVs adds minutes at the start of every Kaggle session (working dir is ephemeral).
- **A1:** Same profile as A0; global stats add a one-time train-pass to fit scalers (cheap).
- **A2 (512):** **Lower** memory (~half the per-window bytes) and **faster** attention — the cheapest ablation to run; cache must be rebuilt for the new window size.
- **A3:** Adds scipy filtfilt/resample over 1951 × ~250k-row files at cold cache (CPU-heavy, several minutes); resample changes row counts → window counts shift.
- **A4:** Noise injection concatenates a second copy of train → ~2× train windows → **~21.8 GB train tensor** → **real OOM risk on Kaggle.** Use the lazy dataset or skip.
- **A5:** Negligible (+0.41% windows).

---

## 9. Recommended FINAL configuration

**A0** with these settings:

- `NUM_EPOCHS = 30` (or 25–40) **+ early stopping on val loss (patience ≈ 6)** — small code addition (§10).
- Keep architecture exactly as-is (D007/D008) — do not touch.
- Keep `CHANNEL_SELECTION = "all"`, `WINDOW_SIZE = 1024`, stride 1024, tail off (change window only via the A2 ablation, not the final run).
- **Normalization:** switch to **per-window z-score** (recommended, FPGA-faithful) — *pending your confirmation in §11*. If you prefer to keep per-file for now, that is acceptable for the software phase but must be re-decided before Stage 6.
- Augmentation off; all filters off.
- `BATCH_SIZE = 64` (verify it still fits and reproduces; cuts val time substantially). Keep `num_workers = 0`.
- Weighted CE stays (auto-applied at ratio 22 > 1.5).
- Save to **versioned** artifact names; run under a **new Kaggle kernel**.

Rationale: this is the smallest change set that converts "a fast test run" into "a converged, reproducible, hardware-comparable software result," without expanding scope or touching the architecture/split.

---

## 10. Exact next implementation steps (after you answer §11)

> No code has been changed yet. These are proposals to confirm first.

1. **`config.py`:** set final `NUM_EPOCHS`, `BATCH_SIZE`; add `EARLY_STOPPING_PATIENCE`; add a `NORMALIZATION_SCOPE = "per_file" | "per_window" | "global_train"` switch (default chosen by your §11 answer). Keep all new flags backward-compatible.
2. **`train_transformer_baseline.py`:** add early-stopping (break when val loss hasn't improved for `patience` epochs); log per-epoch train seconds. The best-checkpoint logic already exists — only the stop condition is new.
3. **Normalization (only if you pick per-window/global):** implement in `prepare_dataset`/`preprocessed_cache`/`make_dataloaders`; fit global stats on the **train split only**; record stats in `experiment_config` for FPGA constants. Bump `CACHE_VERSION` so old per-file caches are not silently reused.
4. **Artifact versioning:** write final outputs as `transformer_final_t4_*` (metrics/history/checkpoint/eval) so historical `D:\` artifacts are preserved.
5. **New Kaggle assets:** create `kaggle/kernel_final/kernel-metadata.json` with a new slug (`fpga-fault-final-train`), `enable_gpu`, `machine_shape NvidiaTeslaT4`; push `src/` as a new **version** of `fpga-fault-src`. A thin runner that calls the real `train()` (full dataset, final epochs) — mirroring the smoke test's import/locate logic.
6. **Smoke-verify** the new kernel on a small `max_files` subset (no architecture/logic change) before the full multi-hour run.
7. Run final training once → evaluate once → freeze the checkpoint for Stage 6.

---

## 11. Questions you must answer before I implement anything

1. **Normalization scope — the key decision (P001):** keep **per-file z-score** (defer FPGA concern), or switch now to **per-window z-score** (recommended, FPGA-faithful), or **global train-fitted**? This determines whether a later retrain is needed.
2. **Epoch budget & early stopping:** OK to set `NUM_EPOCHS ≈ 30` + early stopping (patience ≈ 6)? Any wall-clock ceiling on Kaggle I should respect?
3. **Batch size:** OK to raise `BATCH_SIZE` to 64 (faster, slightly different optimization), or must it stay 8 to match the historical run?
4. **Window length:** keep 1024 for the final run and treat 512 as a separate ablation (A2), or do you want 512 to be the final?
5. **Which ablations (if any)** do you want beyond the required final run — A1, A2, both, or none? (Each is a full Kaggle run.)
6. **Artifact naming:** confirm final outputs should use new versioned names and **not** overwrite `transformer_baseline_*`.
7. **Kaggle:** confirm the final run uses a **new kernel slug** and a **new `fpga-fault-src` dataset version** (smoke-test kernel untouched).
8. **Kaggle RAM:** can you confirm the instance RAM, or should I make the final path use the lazy `WindowedFileDataset` to remove the ~15.6 GB materialization risk?

---

## 12. Is the software/model phase already defensible? What's missing?

**Not yet — but it is close, and the gaps are small and well-scoped.**

**Already defensible:** single clean pipeline; leakage-free **file-level** split; correct 4-class taxonomy; genuinely small Transformer with attention + mean pooling; imbalance handled by weighted CE with strong per-class recall; full P/R/F1/confusion evaluation; reproducibility snapshotting (D012); augmentation/filters correctly off; Kaggle smoke test green on T4.

**Missing before you can honestly say "software/model phase complete":**
1. **A converged final run** — real epoch budget + early stopping, not the 6-epoch "temporary" run that stopped at its best epoch. *(Required.)*
2. **A settled, hardware-reproducible normalization decision** (per-window recommended). *(Required — affects retrain.)*
3. **A clean, versioned final run on Kaggle T4** whose metrics are the official numbers, not overwriting history, under a new kernel slug. *(Required.)*
4. **Verified memory feasibility** of that run (~15.6 GB), or a switch to the lazy dataset. *(Required to de-risk.)*
5. *(Nice-to-have for the report, not blocking):* the window-length 512-vs-1024 note (Stage 2 DoD), per-epoch train timing, optional ROC/AUC.

Once items 1–4 are done and documented, the software/model phase is defensibly complete and you can move to Stage 6 (quantization) with a frozen, reproducible checkpoint.

---

*End of audit. No code was modified and no training/ablation was run, per the constraints. Awaiting answers to §11 before any implementation.*
