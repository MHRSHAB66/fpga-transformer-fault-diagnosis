# Consolidated Results — Evidence-Backed (2026-10-22)

This file is the current single source for project results. It replaces the
obsolete `docs/final_report.md` (2026-09-09 SEQ=1024 snapshot). Every number
below names its **model**, **test type**, **sample count**, **metric** and the
**evidence file** it was read from. Numbers are never carried over from one
model or test to another.

Status labels used throughout:

- **MEASURED-PHYSICAL**: measured on the real MYIR Z-Turn XC7Z020 board.
- **MEASURED-SOFTWARE**: measured by running software (Python / g++ C-model) on real data.
- **ESTIMATE**: tool estimate (e.g. HLS C-synthesis), not a measurement.
- **NOT VERIFIED**: implemented or planned but not yet run at that level.

Class map everywhere: `0 normal · 1 imbalance · 2 misalignment · 3 bearing_fault`.
Test split everywhere: 391 MAFAULDA test **files** → 95,404 windows
(1024 × 8, stride 1024; support 2,440 / 16,348 / 24,156 / 52,460).

---

## 1. Results by model and test (kept separate)

| # | Model | Test type | Samples | Accuracy | Macro-F1 | Status | Evidence |
|---|---|---|---|---|---|---|---|
| A | 2-layer float Transformer (d=32, 4 heads, 18,036 params, SEQ=1024) — **software baseline, not deployed** | full test split, PyTorch | 95,404 | 98.2171 % (93,703) | 97.118 % ¹ | MEASURED-SOFTWARE | `artifacts/runs/2layer_baseline/results/metrics.json` (Kaggle run `20260615_223905_fpga_fault_final_t4`) |
| B | F_1layer SEQ=256 pool4 **float** (1 layer, d=32, 4 heads, ffn=64, 9,492 params) — the model that was quantized for the FPGA | full test split, PyTorch | 95,404 | 97.6489 % (93,161) | 95.4551 % | MEASURED-SOFTWARE | `artifacts/experiments/transformer_f1_seq256_pool4/kaggle_v3/seq256_pool4/summary.json` (`winner_test`) |
| C | Same model, **INT10 Python reference with the export scales** (20-file train calibration = the scales the HLS weights/bitstream were generated from) | full test, on the exact board input file | 95,404 | 97.7684 % (93,275) | 95.6539 % | MEASURED-SOFTWARE | `artifacts/results/board_vs_reference_20261022/comparison.json` (`ref_export_scales`) |
| D | Same float model, **INT10 Python reference with a 12-file calibration** (the previously reported "software INT10 reference") — **a different quantized model**: 12 of 20 activation scales differ from C | full test, on the same input file | 95,404 | 97.7359 % (93,244) | 95.6282 % | MEASURED-SOFTWARE | `artifacts/experiments/transformer_f1_seq256_pool4/int10_ingress_local/combined.json`; reproduced exactly in `comparison.json` (`ref_calib12_scales`) |
| E | **Deployed FPGA**: SEQ=256 INT10 core, dma16 bitstream (sha256 `43e43370…`) | full test, physical board, 2026-10-08 | 95,404 | 97.7684 % (93,275) | 95.6539 % | MEASURED-PHYSICAL | `artifacts/results/fpga_fulltest_20261008_141335/fpga_fulltest_metrics.json` |
| F | Deployed FPGA, same bitstream | full test re-run ×2 (serial + pipelined host loop), physical, 2026-10-22, **logits recorded** | 95,404 each | identical predictions to E (byte-identical `pred.txt`) | — | MEASURED-PHYSICAL | `artifacts/results/fpga_realtime_bench_20261022/board_full_{baseline,pipelined}/` |
| G | Deployed FPGA, live over SSH/Ethernet | **class-balanced** live run, 30 windows/class | 120 | 95.0 % (114) | 94.987 % | MEASURED-PHYSICAL | `artifacts/results/fpga_live_multiclass_20261008_230015/live_evidence.json` |
| H | Deployed FPGA, live over SSH | first 150 contiguous windows (all true class 3, bearing_fault) | 150 | 99.33 % (149) | n/a (single class) | MEASURED-PHYSICAL | `artifacts/results/fpga_live_test_20261008_200509/live_evidence.json` |
| I | Deployed FPGA, live end-to-end (board → SSH → PC parser) | 5 runs of contiguous windows (3,000–10,000) + 2 class-balanced 120-window runs, before/after the runner change | 43,240 total | every received prediction equal to E's for the same window id; class-balanced runs 95.0 % (114/120) as in G | — | MEASURED-PHYSICAL | `artifacts/results/fpga_realtime_bench_20261022/e2e/*/summary.json` |
| J | Historical SEQ=1024 INT10 variant (superseded, never deployed) | full test, Python | 95,404 | 97.11 % (float 97.55 %) | — | MEASURED-SOFTWARE | `kaggle/int10_shard/int10_fulltest_combined.json` |

¹ Not stored in the metrics file; derived from its saved test confusion matrix
(the file stores accuracy, macro-F2 0.97797 and per-class accuracy).

Notes on comparability:

- **G is not comparable to E.** G is a class-balanced 120-window subset; its
  accuracy weights the hard "normal" class 25 % instead of 2.6 %. G's 120
  predictions are all equal to E's predictions for the same window ids.
- **A vs B/E** compares different architectures (2 layers, SEQ=1024 vs 1 layer,
  SEQ=256). A is never the deployed model.
- **C vs D** differ only in quantization calibration; they are different
  quantized models (section 2).

---

## 2. Board output vs software reference

### 2.1 Contract checks (both paths)

| Item | Board path | Python reference path | Same? |
|---|---|---|---|
| Input file | `fpga_testset_in.i32.gz` (int32, row-major `[sample, channel]`, 8,192 words/window) | read from the **same file** (tools below) | yes |
| Window ids | position in the file, 0…95,403; labels from `labels.txt` of the same export | same positions, same `labels.txt` | yes |
| Preprocessing / quantization | done offline once by `tools/export_fpga_testset.py` (per-window z-score → `round(x/0.028626…)`, clip ±511) | the reference consumes the same codes; only the 4:1 aggregation is recomputed | yes |
| 4:1 aggregation | in the IP: `(|s|+2)>>2`, sign restored, clamp | `aggregate_round_div` (same rule) | yes |
| Input scale | 0.028626115354773118 (`model_manifest.json`) | export: identical; calib12: identical | yes |
| Other activation scales | export set (HLS `weights_int10.h`) | export: identical; **calib12: 12 of 20 differ** | **no for D** |
| Output | 4 int32 logits, TLAST on 4th; class = `np.argmax` on the PS | `infer_int` logits; class = `np.argmax` | yes |

### 2.2 Same-input, same-model comparisons (all 95,404 windows)

| Comparison | Level | Agreeing windows | Evidence |
|---|---|---|---|
| Board (2026-10-08) class vs g++ C-model of the deployed HLS source | class | 95,404 / 95,404 | `comparison.json` |
| Board class vs Python reference, export scales (C) | class | 95,404 / 95,404 | `comparison.json` |
| C-model logits vs Python reference (export) logits | all 4 logits | 95,404 / 95,404 | `comparison.json` |
| **Physical board logits (2026-10-22) vs C-model and vs reference (export)** | all 4 logits | **95,404 / 95,404** | `comparison.json` → `board_logits_physical` |

The deployed hardware is therefore bit-exact with the HLS C++ source and with
the Python INT10 reference **that uses the scales the hardware was built from**,
on every test window and down to the logits.

### 2.3 The previously reported +0.03 pp difference (E vs D)

Earlier documents compared E (93,275) with D (93,244) and called it "the FPGA
matches the software reference within +0.03 pp". That was **not a same-model
comparison**: D was produced by
`src/quantize/evaluate_seq256_ingress_int10_shard.py`, which recalibrates on 12
train files, while the HLS export (`src/quantize/export_hls_seq256_int10.py`)
calibrated on 20 files. Same input file, per window:

| | windows |
|---|---|
| board ≠ D (class) | **225** |
| … board correct, D wrong | 124 |
| … D correct, board wrong | 93 |
| … both wrong (different classes) | 8 |
| net difference in correct windows | 31 (= 93,275 − 93,244) |

So an aggregate gap of 31 corresponds to 225 actual prediction differences;
the aggregate alone does not reveal that count. List:
`artifacts/results/board_vs_reference_20261022/disagreements_board_vs_calib12.csv`.

**Cause (confirmed by experiment):** different calibration scales (12 vs 20
files). Running the reference with the export scales removes every difference
(2.2). An alternative hypothesis — Q40 fixed-point requantization in HLS vs
float64 in Python — was tested and **ruled out** (0 logit differences).

**Fix applied:** no hardware or reference-code change was needed. The
documentation now names C as the reference for the deployed hardware and keeps
D as a separate, differently calibrated model. The 12-file evaluator was not
modified (it is an existing evaluation script and its result is still valid
for what it measures).

**Still unknown / not done:** none for correctness on this test set. The live
sensor path (Stage 9) does not exist, so nothing is claimed for sensor data.

### 2.4 Tie policy observation (no change made)

98 test windows (0.10 %) have two equal maximal logits (patterns 1/3: 38,
2/3: 33, 0/2: 20, 1/2: 3, 0/1: 2, 0/3: 2). The approved policy (lowest class
id, `np.argmax`) gets 29 of them right; a highest-id rule would get 67 right.
Choosing a policy from this would be tuning on the test set and would diverge
from the approved reference, so **the policy was not changed**; it is
recorded here only as an observation.

---

## 3. Real-time performance

### 3.1 Requirement

`R_req = f_s / stride = 50,000 Hz / 1,024 = 48.828 windows/s` → one result every
**20.48 ms** (window = stride = 1,024 samples, no overlap; the 4:1 aggregation
is inside the IP and does not change the window rate). Source: `src/config.py`.

### 3.2 Stage by stage

| Stage | Value | Kind | Evidence |
|---|---|---|---|
| Core, HLS C-synthesis interval @100 MHz | 14.905 ms | ESTIMATE | `artifacts/results/seq256_dma_system_dsp_margin_dma16_z7020_20261008/summary.json` |
| Host blocked on DMA+core per window (serial loop, Python timer) | 16.08 ms mean (p95 16.32) | MEASURED-PHYSICAL | `fpga_realtime_bench_20261022/board_full_baseline/summary.json` |
| Input read (gzip of prepared codes) + parse, per window | 2.40 ms mean (serial) | MEASURED-PHYSICAL | same |
| Board-only loop, serial (95,404 windows) | **53.83 inf/s** | MEASURED-PHYSICAL | same |
| Board-only loop, pipelined (95,404 windows) | **60.94 inf/s** | MEASURED-PHYSICAL | `board_full_pipelined/summary.json` |
| End-to-end board → SSH → PC parser, HEAD runner, 2 runs × 10,000 | **51.07 / 52.57 inf/s**; PC inter-arrival p50 18.86 / 18.87 ms; PC parse 36 µs/line | MEASURED-PHYSICAL | `e2e/before_head_runner*/summary.json` |
| End-to-end, overlapping runner, 2 runs × 10,000 (version a, then final version b) | **60.30 / 58.72 inf/s**; inter-arrival p50 16.47 / 16.54 ms; PC parse 67–71 µs/line (includes confidence) | MEASURED-PHYSICAL | `e2e/after_pipelined_runner/`, `e2e/final_runner_sequential10000/` |
| End-to-end, final runner, 3,000 windows | 53.47 inf/s (short run; stalls weigh more) | MEASURED-PHYSICAL | `e2e/final_runner_sequential3000/` |
| Network | not separated: included in the end-to-end figures (overlapping runs: PC inter-arrival p50 ≈ 0.5–0.6 ms above board latency p50) | MEASURED (combined) | same |
| Tk GUI rendering | not measured | NOT VERIFIED | — |

Summary: `artifacts/results/fpga_realtime_bench_20261022/realtime_verdict.json`.
Temperature during all 2026-10-22 runs: 48.8–51.9 °C; independent guard
process (stop at 75 °C) active in every run, no stop occurred.

### 3.3 Bottleneck and change

The serial loop waits ~16 ms for the core and then spends ~2.4 ms reading the
next window. Because the core receives the whole 32,768-byte window in one
MM2S transfer before computing, the input buffer is free once MM2S completes.
`tools/board/fpga_live_runner.py` now reads the next window **while the core
computes** (same bitstream, same single 32,768-byte frame, same window order,
same class decision). Board-only +13.2 % (53.83 → 60.94, one full run each);
end-to-end mean of two 10,000-window runs each 51.82 → 59.51 inf/s (+14.8 %).
Predictions/logits identical before and after. The overlap is applied only to
contiguous runs: with an index file (class-balanced demo) the next read can
gzip-skip thousands of records, which would otherwise be counted in the
per-window latency (first version did this: 1.4 s median); the final version
reads after the result in that mode and reports 16.0 ms again
(`e2e/final_runner_class_balanced120`). Exact runner versions are archived
in the results directory (`fpga_live_runner_HEAD_e42ec4f.py`, `…_20261022a.py`,
`…_20261022b.py` = repository version).

### 3.4 Verdict (what can and cannot be claimed)

- **Sustained average, prepared windows: meets the requirement.** Board-only
  60.94 inf/s over all 95,404 windows and end-to-end 60.30 / 58.72 inf/s over
  10,000 windows exceed 48.83 windows/s (+24.8 %, +23.5 %, +20.3 %). Today's
  serial path also exceeded it (53.83 board-only; 51.07 / 52.57 end-to-end).
  Run-to-run variance is visible (a 3,000-window run: 53.47 inf/s).
- **The 2026-10-08 figure of 46.4 inf/s did not reproduce.** That run was
  ~53 inf/s up to ~41k windows and ~40–41 inf/s afterwards
  (`fpga_run.log`); the same code and data today ran at a flat ~54 inf/s.
  The cause of the 2026-10-08 slowdown is unknown (not identifiable from the
  recorded evidence).
- **Hard per-window deadline: not demonstrated.** In the pipelined full run
  723 of 95,404 iterations took longer than 20.48 ms (96 > 100 ms, max 848 ms,
  mostly input-read stalls). A live stream would need an input FIFO deep
  enough to absorb such stalls; this was not built or tested.
- **Scope:** inputs are INT10 codes prepared offline; acquisition and
  normalization/quantization on the board are not part of the measured path.
- The earlier "live SSH/UI 38.9 inf/s" figure was a 150-window run of 3.86 s
  wall time including startup (`fpga_live_test_20261008_200509`), not a
  sustained rate; it is superseded by I.

---

## 4. Class decision in FPGA logic

**Earlier full-test system:** the core streams 4 logits; the class is
`np.argmax` of those FPGA logits in Python on the Zynq PS
(`pynq/fault_classifier.py`, `FaultClassifier.infer_from_codes`). Tie →
lowest class id. Nothing on the PC classifies.

**Implemented (new, versioned, deployed file untouched):**

- `hls/f1_int10_seq256/vitis/argmax_first_max.h`: synthesizable decision,
  strict `>` scan from class 0 (lowest id wins ties = `np.argmax`).
- `hls/f1_int10_seq256/vitis/transformer_top_seq256_ingress1024_dsp_margin_argmax.cpp`:
  byte-for-byte copy of the deployed top + one AXI-Lite output register
  `pred_class` in the existing `ctrl` bundle.
- **Output contract unchanged (v1):** still exactly 4 int32 logit words,
  TLAST on the 4th; the 16-byte S2MM transfer and every existing driver keep
  working. The class register is additive.
- Driver (`pynq/fault_classifier.py`): looks up `pred_class` /
  `pred_class_ctrl` **in the HWH** (no hard-coded offsets). If present, the
  class is read from the register, the ap_vld bit must be set, and the value
  must equal the argmax of the same logits, otherwise `RuntimeError`.
  If absent (deployed dma16 overlay) the legacy path is used.
  `FaultClassifier.class_source` reports `fpga_register` /
  `ps_argmax_legacy` / `numpy_reference`.
- Flow scripts: `run_seq256_ingress1024_dsp_margin_argmax_verify_and_synth.tcl`
  (Vitis HLS csim + csynth + IP export) and
  `run_seq256_dma_system_dsp_margin_argmax.tcl` (same dma16 system, own
  build directory).

**Verification boundary:**

| Level | Status | Result / evidence |
|---|---|---|
| Unit test of the decision (all tie patterns, INT10 extremes) | MEASURED-SOFTWARE | 3,026 cases, 862 ties, 0 failures (`test_argmax_first_max.cpp`) |
| g++ C-model, golden set incl. ties | MEASURED-SOFTWARE | 40/40 exact (logits + register), 16 tie vectors covering all 6 tie patterns, all 4 classes decided (`golden_ingress1024_argmax/`, built by `tools/make_argmax_golden.py`); deployed top also 40/40 on the same set |
| g++ C-model, full test | MEASURED-SOFTWARE | register = board class on 95,404/95,404; logits identical to the deployed top (`comparison.json`) |
| Driver logic (mocked pynq) | MEASURED-SOFTWARE | `tests/test_hw_class_register.py` (legacy, every class, ties, disagreement, missing valid bit) |
| Vitis HLS C-simulation (ap_int types) | VERIFIED | 40/40 including 16 ties |
| HLS C-synthesis | VERIFIED | interval 1,490,531 cycles; estimates in synthesis report |
| Vivado place-and-route / timing / resources | VERIFIED | WNS +0.250 ns, WHS +0.010 ns; reports contain warnings |
| Bitstream generation | VERIFIED | versioned package `artifacts/deploy/seq256_argmax_z7020_20261022/` |
| Physical board run | VERIFIED | 40/40 golden vectors; full test 95,404/95,404 classes matching earlier board, accuracy 97.768437%; separate balanced live run 39/40 vs truth |

The current live deployment selects the class in FPGA logic. The 2026-10-23
full test of the new overlay and the earlier PS-argmax full test are separate
physical runs. Detailed evidence and the historical hash logging correction are
in `docs/reports/live_guard_argmax_progress_20261022.md`.

---

## 5. Confidence

**Definition:** `CONF = max_c softmax(s_logits · logits_int)[c]`, where
`logits_int` are the 4 integers produced by the FPGA core and
`s_logits = 0.10779884107882078` is the deployed model's exported logits scale
(`artifacts/deploy/seq256_z7020_pynq_dma16_py36_20261008/config/integer_config_scales.json`).
It is the model's own softmax probability of the class the FPGA chose. It never
changes the class. Nothing is fitted (no temperature scaling, no thresholds).

> **Update 2026-10-24:** the Live display now uses a temperature-scaled CONF,
> T = 1.4248 fitted on the validation split only
> (`artifacts/calibration/conf_temperature_v1/`): test ECE 0.79 % → 0.09 %, NLL
> 0.0628 → 0.0584, classes unchanged. Details:
> `docs/reports/hard_rt_calibration_power_20261024.md`; reliability diagram:
> `docs/figures/final/reliability_confidence.png`. The uncalibrated numbers below
> remain correct for the raw softmax.
Implementation: `host/confidence.py` (PC side), from logits that the live
runner now emits as a trailing `logits=a,b,c,d` field.

**Measured calibration** (`artifacts/results/board_vs_reference_20261022/confidence/confidence_calibration.json`;
the logits are bit-exact with the board):

| Split | Windows | Accuracy | Mean CONF | ECE (15 bins) | Brier | Acc. at CONF ≥ 0.9 (coverage) |
|---|---|---|---|---|---|---|
| validation | 47,580 | 98.10 % | 98.81 % | 0.71 % | 0.0291 | 99.29 % (96.4 %) |
| test | 95,404 | 97.77 % | 98.55 % | 0.79 % | 0.0336 | 99.24 % (95.5 %) |

**Limitations:** uncalibrated; in the 0.6–0.93 range it overestimates
accuracy by roughly 5–12 points on both splits; windows from the same file are
correlated, so the bin statistics are not independent samples; tied windows
get CONF ≤ 0.5; displayed with 3 decimals, so values ≥ 0.9995 show as
`1.000`. Valid only for this model/scale.

**Where it is shown:** live mode only, when the runner sends logits (the new
runner does). Replay of the 2026-10-08 full test keeps **CONF=N/A** because
that run recorded classes only. An older runner without logits → CONF=NA.
End-to-end check on the board: every CONF value received on the PC equals the
value computed from the reference logits for that window id — 10,000/10,000
(`e2e/after_pipelined_runner`), 10,000/10,000 (`e2e/final_runner_sequential10000`),
120/120 class-balanced (`e2e/final_runner_class_balanced120`).

---

## 6. Hardware implementation numbers (deployed dma16 build)

| Item | Value | Kind | Evidence |
|---|---|---|---|
| Part / clock | xc7z020clg400-1, 100 MHz FCLK0 | — | `seq256_dma_system_dsp_margin_dma16_z7020_20261008/summary.json` |
| Timing | WNS +0.364 ns, TNS 0, WHS +0.018 ns — met | MEASURED (post-route report) | `…/vivado_reports/post_route_timing_summary.rpt` |
| LUT / FF / BRAM tiles / DSP | 31,734 (59.65 %) / 30,921 (29.06 %) / 70.5 (50.36 %) / 210 of 220 | MEASURED (post-route report) | `…/vivado_reports/post_route_utilization.rpt` |
| HLS estimate | interval 1,490,527 cycles (14.905 ms), DSP 156, LUT 52,864 | ESTIMATE | `…/hls_reports/transformer_top_csynth.rpt` |
| Bitstream on board | sha256 `43e43370…2d2c2`, same as package | MEASURED (hash) | board + package |

Earlier documents quoted WNS +0.036 ns / 31,687 LUT: those belong to the
previous 16 KB-DMA build (`seq256_dma_system_dsp_margin_z7020/`), which is
**not** the deployed bitstream.

---

## 7. Reproduce

```bash
# C-model of the deployed top over the board's exact input (g++, ~2 min on 12 shards)
g++ -O2 -std=c++17 -Wno-unknown-pragmas -I hls/f1_int10/gpp_csim/stubs -I hls/f1_int10_seq256/vitis \
  hls/f1_int10_seq256/vitis/transformer_top_seq256_ingress1024_dsp_margin.cpp \
  hls/f1_int10_seq256/gpp_csim/batch_logits.cpp -o tmp/csim/batch_dsp_margin.exe
python -m tools.run_csim_fulltest --exe tmp/csim/batch_dsp_margin.exe \
  --input artifacts/results/fpga_fulltest_20261008_141335/fpga_testset_in.i32.gz \
  --n 95404 --shards 12 --out <new_dir>/csim_dsp_margin
# argmax variant: add -DWITH_PRED_CLASS and the _argmax.cpp top
# Python INT10 reference on the same file (export / calib12 scales)
python -m tools.run_numpy_reference_fulltest --input <same .i32.gz> --n 95404 --scales export --out <new_dir>/numpy_reference
python -m tools.run_numpy_reference_fulltest --input <same .i32.gz> --n 95404 --scales calib12 --out <new_dir>/numpy_reference
python -m tools.compare_board_vs_reference --dir <new_dir> \
  --board-logits artifacts/results/fpga_realtime_bench_20261022/board_full_pipelined/logits_i32.bin
# argmax tests
g++ -std=c++17 -I hls/f1_int10_seq256/vitis hls/f1_int10_seq256/vitis/test_argmax_first_max.cpp -o t && ./t
g++ -O2 -std=c++17 -Wno-unknown-pragmas -I hls/f1_int10/gpp_csim/stubs -I hls/f1_int10_seq256/vitis \
  hls/f1_int10_seq256/vitis/transformer_top_seq256_ingress1024_dsp_margin_argmax.cpp \
  hls/f1_int10_seq256/vitis/testbench_ingress1024_argmax.cpp -o tb && ./tb hls/f1_int10_seq256/golden_ingress1024_argmax
# confidence calibration (validation + test)
python -m tools.evaluate_confidence_calibration --test-logits <new_dir>/numpy_reference/numpy_export_logits.txt --out <new_dir>/confidence
# board (needs board + permission): tools/board/fpga_stage_bench.py + temp_guard.sh (see bench_run.sh in
# artifacts/results/fpga_realtime_bench_20261022/), PC end-to-end: python -m tools.measure_live_e2e ...
python -m unittest discover -s tests
```

---

## 8. ROC/AUC and final figures (added 2026-11-09)

ROC/AUC is listed in the proposal as an optional metric ("if possible"); it is
now computed for the **deployed** model from the 4 physical board logits of all
95,404 test windows (`fpga_realtime_bench_20261022/board_full_pipelined/logits_i32.bin`,
bit-exact with the INT10 reference, section 2). Score = softmax(s_logits ·
logits), uncalibrated; one class vs rest. MEASURED-SOFTWARE on physical outputs.

| Class | AUC (one vs rest) | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| normal | 0.9990 | 92.00 % | 87.17 % | 89.52 % | 2,440 |
| imbalance | 0.9991 | 96.15 % | 97.77 % | 96.96 % | 16,348 |
| misalignment | 0.9994 | 96.36 % | 99.10 % | 97.71 % | 24,156 |
| bearing_fault | 0.9990 | 99.23 % | 97.65 % | 98.43 % | 52,460 |
| **macro** | **0.9991** (micro 0.9993) | 95.93 % | 95.42 % | 95.65 % | 95,404 |

Normal is the weakest class: 285 of its 2,440 windows (11.7 %) are predicted as
misalignment.

Figures (PNG + PDF), all drawn from recorded evidence by
`python -m tools.plot_final_figures`; every plotted number is in
`artifacts/results/final_figures_20261109/figure_data.json`:

| Figure | Content |
|---|---|
| `docs/figures/final/roc_ovr_board` | ROC per class, log FPR axis |
| `docs/figures/final/confusion_matrix_board` | board predictions vs labels |
| `docs/figures/final/reliability_confidence` | CONF calibration before/after T |
| `docs/figures/final/rt_response_ccdf` | hard real-time response times vs the 20.48 ms deadline |
| `docs/figures/final/e2e_latency_profile` | TCP end-to-end mode: board stage vs GUI render |
| `docs/figures/final/throughput_vs_requirement` | measured windows/s vs 48.83 /s |
| `docs/figures/final/fpga_resources` | post-route utilisation of the argmax build |
| `docs/figures/final/cpu_vs_fpga_latency` | same INT10 model per window: board ARM, FPGA (both cores), laptop CPU (added 2026-11-10) |

---

## 9. Results added 2026-11-09 (evening, physical board)

| Item | Result | Kind | Evidence |
|---|---|---|---|
| Raw-input core (preprocessing in the FPGA), full test | 97.7705 % (93,277 / 95,404); logits = C-model on 95,404 / 95,404 | MEASURED-PHYSICAL | `docs/reports/onchip_preprocessing_20261109.md` |
| Raw-input core, golden | 41 / 41 exact incl. 8 ties, 4 edge windows | MEASURED-PHYSICAL | same |
| Raw-input core, hard real-time (R1 configuration) | 0 misses in 30,000, p50 15.96 ms, max 16.64 ms | MEASURED-PHYSICAL | same |
| Raw-input core, post-route | WNS +0.167 ns; LUT 36,670, FF 36,163, BRAM 76.5, DSP 210 | MEASURED (post-route) | same |
| UART result path | 10,000 / 10,000 frames byte-identical, 0 malformed / gaps / duplicates, 48.80 frames/s | MEASURED-PHYSICAL | `docs/reports/uart_result_path_20261109.md` |
| Same INT10 model on the board ARM (one core) | p50 218.6 ms per window -> FPGA ~14x faster | MEASURED-PHYSICAL | `docs/reports/cpu_baseline_20261109.md` |
| Same model on a laptop i7-13620H (reference) | p50 6.6-8.0 ms per window | MEASURED-SOFTWARE | same |
