# Repository cleanup plan — SEQ=256 pre-board snapshot (2026-09-30)

**Non-destructive recommendation only.** Nothing here has been deleted, moved, or
committed. It classifies the current untracked/temporary items into
**KEEP (track)**, **ARCHIVE (historical, keep)**, and **IGNORE (reproducible
temp)**. Pre-existing tracked modifications by the user are left untouched.

## IGNORE — reproducible / temporary (added to `.gitignore` this session)

Already ignored before: `/tmp/`, `*.log`. Added now:

| Pattern | Why |
|---|---|
| `*.jou` | Vivado journal files (root `vivado*.jou`, `*_backup.jou`) — regenerated every run |
| `.Xil/` | Vivado transient working dir |
| `/seq256_ingress1024_dsp_margin/` | Root Vitis HLS project build dir — regenerate from `hls/f1_int10_seq256/vitis/*.tcl` + sources |

`/tmp/seq256_dma_system_dsp_margin/` (the full Vivado project, ~hundreds of MB of
`.cache/.gen/.runs/.hw`) is already ignored via `/tmp/`. Its canonical outputs
(bit/hwh/xsa + reports) are copied into the tracked package below, so nothing of
value is lost when `/tmp/` is cleaned.

## KEEP — canonical final artifacts (recommend track)

| Path | Role |
|---|---|
| `artifacts/results/seq256_dma_system_dsp_margin_z7020/` | **Final evidence package** (HLS src, CSim, reports, bit/hwh/xsa, manifest, summary) |
| `artifacts/deploy/seq256_z7020_pynq/` | **Deployment bundle** (overlay + driver + golden + README) |
| `artifacts/results/seq256_dma_system_z7020/` | Prior 220-DSP build — retained as the DSP-margin comparison baseline |
| `artifacts/results/hls_seq256_int10_summary.json` | SEQ=256 machine-readable summary (updated) |
| `artifacts/results/hls_seq256_int10_z7020/`, `..._z7010/`, `..._ingress1024_z7020/`, `..._ingress1024_z7020_postroute/` | SEQ=256 HLS/OOC evidence |
| `artifacts/experiments/transformer_f1_seq256_pool4/` | Checkpoint + `int10_ingress_local/combined.json` (accuracy source of truth) |
| `artifacts/quantized/f_seq256_int10/integer_config.json` | Calibrated INT10 scales (drives the NumPy backend) |
| `hls/f1_int10_seq256/` | HLS sources, weights header, golden vectors, Tcl |
| `src/quantize/*`, `src/data/token_aggregation.py` | INT10 model + aggregation |
| `tests/test_frame_protocol.py`, `tests/test_pynq_seq256_ingress.py`, `tests/test_token_aggregation.py` | New/near-new tests |
| `docs/reports/hls_seq256_int10_results.md`, `docs/reports/seq256_preboard_readiness.md` | Updated report + readiness checklist |

**Large-binary note:** the `.bit` (~4 MB) appears in the package, the deploy
bundle, and the prior-build dir; `.xsa` (~1.2 MB) in two. This is intentional
(deliverable + evidence + baseline) but is ~10 MB of duplicated binaries. Options
if repo size matters: keep only the deploy bundle's copy in git and reference the
package by hash, or use Git LFS for `*.bit`/`*.xsa`. **Decision deferred to the
user; no change made.**

## ARCHIVE — historical experiments (keep, do NOT delete)

These are measured experiment results (AGENTS.md: never delete artifacts without
approval; preserve historical experiments). They are safe to keep untracked, to
track as-is, or to consolidate under an `archive/`/`historical/` subtree later —
but their measured contents must not be rewritten.

- Real-time HLS variants: `artifacts/results/hls_rt3a_kv2/`, `hls_rt5_splitbuf/`,
  `hls_rt_attn2_ff4/`, `hls_rt_combined_kv2_df/`, `hls_rt_dataflow2/`,
  `hls_rt_dataflow_FAILED/`, `hls_rt_df_2head/`, `hls_rt_explicit_2head/`,
  `hls_rt_explicit_4head/`, `hls_rt_kv_dataflow/`, `hls_rt_prefix_opt1/`,
  `hls_rt_prefix_opt2/`, `hls_rt_t1_kbank/`, `hls_rt_t2_reqshare/`.
- Z7010 fitting campaign: `hls_z7010_opt_a..e/`, `hls_z7010_opt_ctx_v3_bram_fixed/`.
- Z7020 context opt: `hls_z7020_opt_ctx_v1_aggressive_inline/`,
  `..._v2_noinline_HANG/`, `..._v3_bram_fixed/`.
- Report figures / summaries: `weekly11_*`, `hls_throughput_study*`,
  `hls_z7010_fitting_campaign*`, `hls_softmax_*`, `hls_realtime_phase2*`,
  `hls_realtime_continuation_summary.json`, `hls_attention_bottleneck_analysis.json`,
  `hls_measured_opt_vs_baseline.png`, `hls_bottleneck_optimization.png`.

## Pre-existing user modifications — DO NOT TOUCH

`src/data/make_dataloaders.py`, `src/train_transformer_baseline.py`,
`kaggle/code_dataset/src/data/make_dataloaders.py` were modified before this
session and are unrelated to it. `docs/project_status.md`, `docs/session_notes.md`
and `pynq/*` were updated this session (documentation reconciliation).

## Suggested next manual steps (not performed here)

1. Review the KEEP list and `git add` the canonical package/deploy/docs/tests.
2. Decide on the large-binary policy (in-git vs LFS vs external).
3. Optionally relocate ARCHIVE dirs under a `historical/` subtree in one commit.
4. Clean `/tmp/`, `.Xil/`, `*.jou` freely — all reproducible.
