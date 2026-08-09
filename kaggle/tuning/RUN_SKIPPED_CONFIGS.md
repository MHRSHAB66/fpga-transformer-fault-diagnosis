# Running the 3 skipped tuning configs on Kaggle

The first tuning session (`tuning_20260616_161108`) completed **F_1layer, A_baseline,
B_wide_ffn** and skipped **E_3layers, C_d48, D_d64_slow** when the 8 h budget ran out.

This kernel is now set to run exactly those three, with the **same pipeline, fixed
settings and metric definitions** as the first run, so the results are directly
comparable.

## The three configs
| name | d_model | layers | ffn | lr | note |
|------|--------:|------:|----:|----|------|
| E_3layers  | 32 | 3 | 64  | 1e-3 | depth 2→3 |
| C_d48      | 48 | 2 | 96  | 1e-3 | wider embedding (non-pow2) |
| D_d64_slow | 64 | 2 | 128 | 5e-4 | largest; upper bound |

Fixed (unchanged): window 1024 · stride 1024 · per-window z-score · batch 64 ·
max 12 epochs · patience 3 · seed 42.

## What runs
`RUN_ONLY_DEFAULT = ["E_3layers", "C_d48", "D_d64_slow"]` (top of
`fpga-fault-tuning.py`). Override at runtime with env `TUNING_RUN_ONLY`
(comma-separated) — e.g. to finish a single leftover config: `TUNING_RUN_ONLY=D_d64_slow`.

## Option A — Kaggle CLI (from this repo)
Needs your API token at `~/.kaggle/kaggle.json` (Kaggle → Account → Create New API Token).

```bash
# push the updated kernel and run it (GPU T4, datasets already wired in kernel-metadata.json)
kaggle kernels push -p kaggle/tuning

# check progress
kaggle kernels status mehrdadsheikhabbasi/fpga-fault-tuning

# when complete, download outputs
kaggle kernels output mehrdadsheikhabbasi/fpga-fault-tuning -p kaggle/tuning/output_skipped
```

## Option B — Kaggle web UI
1. Open the kernel **mehrdadsheikhabbasi/fpga-fault-tuning** (or create a notebook and paste
   `fpga-fault-tuning.py`).
2. Attach datasets: **mehrdadsheikhabbasi/fpga-fault-src** (code) and **vuxuancu/mafaulda-full** (data).
3. Accelerator = **GPU T4**, Internet = off.
4. **Save Version → Save & Run All (Commit)**.

## Time / sessions (important)
These three are the heavy configs. On a single T4 they may **not all fit in one 9 h
session** — the built-in budget will finish as many as it can and mark the rest
`skipped`. That is safe: each config's results are saved the moment it finishes.

- Recommended if worried about the cap: run **one config per session** by setting
  `TUNING_RUN_ONLY` to a single name (`E_3layers`, then `C_d48`, then `D_d64_slow`).
- Or run all three; if one is skipped, start a new session with `TUNING_RUN_ONLY`
  set to just the leftover name.

## Output files to bring back
From `/kaggle/working/` (downloadable):
- `tuning_final_report.json` (copied to the working root for convenience)
- `tuning_<timestamp>/<config>/results/result.json` — one per completed config
- `tuning_<timestamp>/tuning_partial_summary.json`

Send me any of these (the per-config `result.json` files are enough) and I will:
1. merge them with F/A/B into a complete 6-config comparison, and
2. update the deck's tuning slide (slide 9) — replacing "planned but not executed" with
   the measured results.
