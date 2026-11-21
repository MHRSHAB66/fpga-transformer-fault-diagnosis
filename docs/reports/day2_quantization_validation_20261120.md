# Day 2 — INT10 numeric-contract tests and calibration sensitivity (2026-11-20)

Scope: the deployed SEQ=256 / pool4 / INT10 model (`hls/f1_int10_seq256`, dma16
bitstream). **Nothing deployed was changed**: checkpoint, INT10 weights, scales
(`artifacts/quantized/f_seq256_int10/integer_config.json`), HLS sources, generated
header, golden vectors and bitstreams are untouched. This is validation only.

Evidence directory: `artifacts/experiments/day2_calibration_sensitivity_20261120/`.

---

## 1. Numeric-contract tests

`tests/test_int10_numeric_contract.py` (31 tests). It compares three things that are
kept separate:

| Name | What it is |
|---|---|
| **oracle** | `src/quantize/int10_contract.py`: an exact Python-`int` copy of the deployed HLS arithmetic (signed Q40 coefficients + `div_round_even`), independent of bit width |
| **hls** | the deployed top `transformer_top_seq256_ingress1024_dsp_margin.cpp`, `#include`d verbatim by `tests/cpp/int10_contract_harness.cpp` and built with g++ in **two builds**: native wide integers (the C-model build) and `-DHLS_USE_AP_INT` (`ap_int<10>/<32>/<20>`, the synthesis widths; Xilinx headers from `G:/Xilinx/Vitis_HLS/2020.2/include`) |
| **reference** | `IntegerTransformer` / `integer_ops` (float64 requant), the source of every reported accuracy |

"Bit-exact" means the integers are equal. "Approximation" means the result is
compared with real-valued math within a stated tolerance.

### Rounding policies of the deployed system

| Step | Rule | Implementation |
|---|---|---|
| Ingress quantisation | ties to **even**, then clip to ±511 | `np.round(x/s_in)` on the PC side |
| 4:1 aggregation | ties **away from zero**, then clamp | `sign(s)·((|s|+2)>>2)` (HLS ingress loop, `aggregate_round_div`) |
| Requant, residual, LN affine, LN mean/var | ties to **even** | Q40: `div_round_even(acc·Mq, 2^40)`, `Mq = round_half_even(Decimal(M)·2^40)` |
| Softmax normalisation | ties up (num ≥ 0) | `(num·511 + denom/2) / denom` |
| exp / GELU / rsqrt | ROM lookups, no rounding at run time | integer index, clamped |

The legacy `quantize_multiplier`/`apply_multiplier` (M0·2^-31·2^shift, ties away
from zero) is **not** used by the deployed core.

### Results (all PASS)

| Area | Tests | Result |
|---|---|---|
| A. INT10 quantisation | range ±511 (never −512), ties (0.5→0, 1.5→2, 510.5→510, 511.5→511), near-zero, calibration max → 511, saturation ≠ wrap | bit-exact |
| B. Q40 requant | hand-built ties; 3,000 random `div`; signed `mul_s64_shift_add`; every deployed multiplier at ±1/±2 around each rounding boundary of outputs 0…600, including saturation; synthetic exact ties and **negative** multipliers | hls (both builds) == oracle |
| B. Header provenance | every `*_q40` array in `weights_int10.h` == `round_half_even(Decimal(data text)·2^40)`; the double arrays equal the data text | bit-exact |
| B. Python reference vs Q40 | **all reachable accumulators** for all 12 single multipliers (~173 M values); residual adds exhaustive (1023² pairs); LayerNorm 15,000 random rows (5,000 per LN) | **0 mismatches** |
| B. Widths | all MAC accumulators < 2^31 (`ap_int<32>`, and `long` on LLP64); `acc·Mq`, residual and `c·inv·gm+bt` < 2^63; LN sum of squares < 2^31; exp entries < 2^19 (`ap_int<20>`); softmax numerator < 2^31 | bounds hold |
| C. GELU | LUT rebuilt from the export scales == header (both tables); within 0.5 code of erf-GELU wherever representable; clipped only at the top; clamp at ±511/±2^20 | bit-exact + approximation |
| C. exp / softmax | table == rebuild, exp(0)=2^15, entry 0 underflows to 0, monotone, ≤0.5 code from exp; softmax oracle == reference on 106 rows (equal, extreme, random); ≤2/511 from float softmax | bit-exact + approximation |
| C. Attention head | `df_head<0>` (scores → softmax → context) on 11 random/extreme cases | hls == oracle |
| C. rsqrt / LayerNorm | rsqrt for v = −3…69,999, 2^e±1, 5,000 random up to 2^31; float-log2 reference == loop oracle; LN rows with zero/tiny/maximum variance on all 3 LNs | bit-exact; <1 % rel. error for v < 2^16 |
| D. Aggregation | every reachable group sum (−2044…2044); ties ±2, ±6, ±10 differ from half-even; boundaries ±511 | bit-exact |
| D. End-to-end | `transformer_top` (includes the inline aggregation) vs the reference on 5 tie-heavy windows (>1,000 tie groups) plus the 16 ingress golden vectors | logits identical |

Findings from these tests (none is a bug in the deployed arithmetic):

1. With the deployed constants, the float64 reference is bit-identical to the Q40
   hardware contract over the **entire reachable input range**, not just on the
   95,404 test windows. This is a property of these constants, not of the two
   algorithms: `test_float_reference_is_not_q40_in_general` shows a multiplier
   (0.5 + 2^-42) where the two disagree. A re-export with new scales should be
   re-checked with these tests.
2. `ln_epsq` rounds to **0** for all three LayerNorms (`1e-5/s_in²` < 0.5). Zero
   variance is guarded only by `rsqrt_fxp`'s `v<1 → 1`. Tested: a constant row
   returns `round(beta)`.
3. The softmax of 256 equal scores gives 2 per entry (total 512, not 511). This is
   bounded rounding and matches the reference exactly.
4. `requant(long acc, …)` takes `long`, which is 32 bits on Windows/LLP64. This is
   safe today because every accumulator is < 2^31 (`WidthBoundTests`), but it would
   become a portability hazard if the model grows.
5. The `integer_ops` self-test covers only int16 / GELU-16 toy settings; INT10
   coverage is now in the new test file.

---

## 2. Calibration-size sensitivity

### Method

- **Calibration (train split only, local CPU):** for seeds {42, 1, 2} × sizes
  {12, 20, 40, 80}: `train_df.sample(n, random_state=seed)`, the exact rule of the
  export. For a fixed seed, pandas draws a prefix of one permutation, so the sets
  are **nested** (12 ⊂ 20 ⊂ 40 ⊂ 80; asserted). Scales come from the approved float
  path (1024-sample z-score → float 4:1 mean → `collect` → max-abs/511).
  - `s42_n20` reproduces `integer_config.json` **bit-for-bit** (4,880 windows) = canonical.
  - `s42_n12` reproduces the historical calib12 scales bit-for-bit
    (`numpy_calib12_scales.json`, the 97.7359 % row D).
  - The file lists are in `calibration_sets.json`.
- **Evaluation (validation split, 195 files / 47,580 windows; never used for
  calibration):** 1024 normalised samples → INT10 quantise → `aggregate_round_div`
  4:1 → 256-token `IntegerTransformer`. This is the hardware ingress order, not
  aggregate-then-quantise.
  - An instrumented subclass counts **pre-clip** saturation at every requant
    position. Its logits were asserted identical to the stock `infer_int` on the
    first file of every worker.
  - The float model (float mean → float path) is the reference row.
  - Uncertainty: paired **file-level** bootstrap (2,000 resamples) of the
    difference from canonical.
- **80 files:** added because the scales had not converged at 40 files (§2.2).
- **Test set: NOT RUN.** Nothing on validation justified it (no difference from
  canonical is resolvable), and the 12-vs-20 test comparison already exists
  (consolidated rows C/D).

### 2.1 Accuracy (validation)

| Config | Calib. windows | Acc. % | Macro-F1 % | Δacc vs canon., 95 % CI (pp) | Normal recall | Changed preds (fixed/broken) |
|---|---:|---:|---:|---|---:|---|
| float reference | — | 97.976 | 96.342 | [−0.210, −0.036] | 0.903 | 154 (45/103) |
| s42_n12 (historical 12) | 2,928 | 98.060 | 96.379 | [−0.101, +0.017] | 0.890 | 103 (40/58) |
| **s42_n20 (deployed)** | 4,880 | **98.098** | **96.476** | — | 0.898 | 0 |
| s42_n40 | 9,760 | 98.100 | 96.495 | [−0.029, +0.036] | 0.899 | 94 (46/45) |
| s42_n80 | 19,520 | 98.134 | 96.460 | [−0.027, +0.107] | 0.901 | 168 (88/71) |
| s1_n12 | 2,928 | 98.146 | 96.430 | [−0.048, +0.162] | 0.912 | 178 (94/71) |
| s1_n20 | 4,880 | 98.148 | 96.431 | [−0.053, +0.166] | 0.912 | 183 (97/73) |
| s1_n40 | 9,760 | 98.157 | 96.453 | [−0.036, +0.168] | 0.911 | 180 (98/70) |
| s1_n80 | 19,520 | 98.153 | 96.484 | [−0.050, +0.181] | 0.917 | 180 (97/71) |
| s2_n12 | 2,928 | 98.129 | 96.508 | [−0.017, +0.084] | 0.899 | 135 (73/58) |
| s2_n20 | 4,880 | 98.081 | 96.404 | [−0.057, +0.021] | 0.897 | 133 (60/68) |
| s2_n40 | 9,760 | 98.167 | 96.383 | [−0.048, +0.189] | 0.910 | 212 (116/83) |
| s2_n80 | 19,520 | 98.159 | 96.463 | [−0.048, +0.179] | 0.914 | 198 (109/80) |

Per class (windows: normal 1,220 / imbalance 8,052 / misalignment 12,200 /
bearing 26,108), recall ranges across all 12 INT10 configs are:

- normal 0.890–0.917
- imbalance 0.995–0.997
- misalignment 0.994–0.997
- bearing 0.973–0.975

All macro-F1 CIs include 0. Seed-averaged accuracy by size is 98.112 / 98.109 /
98.141 / 98.148 % for 12 / 20 / 40 / 80 files, a trend of about +0.04 pp that is
smaller than any single CI.

The normal class has only **5 validation files**. Its recall differences
(≤ 24 windows) are concentrated in two of them (`normal/14.336.csv`,
`normal/32.9728.csv`), so they are not interpretable as class-level effects.

Changed predictions number 94–212 (0.20–0.45 % of windows), spread over 37–54
files. These are 0.2–0.45 % disagreements between *different quantized models*,
not accuracy changes.

### 2.2 Scales

Max-abs calibration is driven by rare outlier windows. Largest relative change
across all draws versus canonical:

| Position | max \|Δscale\|/scale | Position | max \|Δscale\|/scale |
|---|---:|---|---:|
| l0.scores | **+175 %** | l0.ctx | +65 % |
| l0.attn_out | +73 % | embed | +63 % |
| l0.res1 | +73 % | l0.v | +62 % |
| l0.q | +67 % | l0.k | +44 % |

`input` changes by at most 4.5 % and the tail positions by 4–30 %. Scales over
nested sets can only grow. Even at 80 files the three seeds still differ (e.g.
`l0.scores` 0.521 / 0.548 / 0.629 versus canonical 0.229), so max-abs on this
data has **no stable limit at these sizes**. The deployed seed-42 draw simply
contains no attention outlier until file 41–80. Full table: `scales_vs_canonical.csv`.

### 2.3 Saturation (pre-clip, validation)

In ppm of values / windows with any saturation at that position (selected rows;
full table `val_saturation.csv`):

| Position | s42_n12 | **s42_n20** | s42_n40 | s42_n80 | s1_n20 | s2_n20 | s2_n80 |
|---|---|---|---|---|---|---|---|
| input | 4.70 / 1,344 | 4.70 / 1,344 | 4.62 / 1,341 | 4.43 / 1,323 | 4.54 / 1,332 | 4.57 / 1,337 | 4.44 / 1,323 |
| l0.ctx | **164 / 640** | 0.54 / 16 | 0.62 / 16 | 0 | 0 | 0 | 0 |
| l0.attn_out | **171 / 780** | 0.56 / 24 | 0.62 / 27 | 0 | 0 | 0 | 0 |
| pool | 657 / 999 | **675 / 1,027** | 94 / 143 | 16 / 24 | 78 / 118 | 118 / 179 | 5.9 / 9 |
| cls1 | 88 / 67 | 89 / 68 | 13 / 10 | 2.6 / 2 | 7.9 / 6 | 188 / 143 | 0 |
| logits | 242 / 46 | 21 / 4 | 0 | 0 | 215 / 41 | 962 / 183 | 11 / 2 |

- The deployed scales saturate the attention path at ≤ 0.6 ppm. Their largest
  saturation is at **`pool`**: 2.2 % of windows have at least one of 32 pooled
  channels clipped. This falls with 40–80 files without any measurable change in
  accuracy.
- The historical 12-file scales saturate attention about 300× more often, which
  explains the 225 board-vs-12-file differences recorded earlier.
- Counting outputs equal to ±511 overestimates saturation. For canonical:
  `l0.ctx` 261 at the limit vs 211 truly clipped, `l0.res1` 90 vs 57, and
  `cls_act` 73 vs **0** (the GELU LUT reaches 511 exactly without clipping).

### 2.4 Cost

| Step | Where | Time | Peak memory |
|---|---|---|---|
| Calibration | local, 4 processes | 30–53 s for 12–20 files, 85–131 s for 40, 209–413 s for 80 (`collect`, per config, with contention); 11.6 min wall for all 12 | 523 / 540 / 605 / 760 MB RSS per worker for 12 / 20 / 40 / 80 files (includes torch) |
| Evaluation | Kaggle CPU, 4 shards × 4 processes | ≈ 1,640 CPU-s per INT10 config over 47,580 windows (≈ 29 windows/s per process); 826–1,579 s wall per shard | — |

The workload is CPU-bound NumPy (int64 matmul), so no GPU was requested.

### 2.5 Answers

1. **Is 20 files sufficient?** Yes for accuracy. No 12/40/80-file draw from any
   seed differs from the deployed model by a resolvable amount (all 95 % CIs
   include 0). The *scales* are not stable at any size (max-abs is
   outlier-driven), but INT10 accuracy is insensitive to that.
2. **More files?** No material change: about +0.04 pp mean accuracy, within the CIs.
3. **Saturation?** Larger sets reduce tail saturation (`pool`, `cls1`) and remove
   the residual attention saturation. Smaller sets (12, seed 42) increase
   attention saturation sharply. Neither moves accuracy measurably on validation.
4. **Most sensitive positions:** scales at `l0.scores`, `l0.attn_out`/`l0.res1`,
   `l0.q`, `l0.ctx`; saturation at `pool`, `cls1`, `logits`.
5. **Class regressions?** None that can be resolved. Normal-class differences are
   ≤ 24 windows from 2 of only 5 files.
6. **Keep the deployed 20-file configuration?** **Yes, unchanged.** No
   alternative is shown to be better. If a re-export happens for other reasons,
   a larger calibration set (e.g. `s42_n80`, less `pool` saturation) is a
   *candidate* only. It would need its own validation, a new HLS export, C-model
   and board verification. A percentile-based calibration would address the
   outlier sensitivity directly; this was not tested.

Limitation: the accuracy numbers use the float64 reference. Its bit-equality with
the Q40 hardware was proven for the deployed constants (§1). For the 11
alternative scale sets it is expected but not verified, because no HLS export was
made for them.

---

## 3. Documentation corrections (2026-11-20)

- `src/quantize/export_hls_seq256_int10.py` header: 97.11 % corrected to the
  deployed 97.7684 %, with a note that 97.11 % is the historical SEQ=1024 INT10
  variant. Also corrected the output paths, the run command, and the claim that it
  remaps state_dict keys (it does not).
- `src/quantize/integer_ops.py`: legacy M0/shift vs the deployed Q40, and the
  scope of the int16 self-test.
- `src/quantize/integer_model.py`: the integer-tensor reference uses float64
  constants and float64 multiply-and-round, while the deployed HLS uses Q40 fixed
  point; widths are overridden to 10 bits by `set_bits(10)`.
- All three are docstring-only edits; behaviour is identical.

---

## 4. Reproduce

```bash
# numeric-contract tests (~45 s; g++ needed for the hls tests, Xilinx headers for the ap_int build)
python -m pytest tests/test_int10_numeric_contract.py -q

# calibration (train only, ~12 min on 4 processes)
python -m src.quantize.calibration_sensitivity calibrate --seeds 42 1 2 --sizes 12 20 40 80 \
    --workers 4 --out artifacts/experiments/day2_calibration_sensitivity_20261120

# validation evaluation: Kaggle (4 private CPU kernels) ...
python kaggle/day2_calib_sensitivity/build_kernels.py
kaggle kernels push -p kaggle/day2_calib_sensitivity/shard0     # ... shard1..3
kaggle kernels output mehrdadsheikhabbasi/fpga-fault-day2-calib-sens-shard-0 -p <dir>
# ... or locally: --shard i --num-shards 4 (about 1 CPU-hour per shard)
python -m src.quantize.calibration_sensitivity evaluate \
    --scales artifacts/experiments/day2_calibration_sensitivity_20261120/calibration_sets.json \
    --shard 0 --num-shards 4 --workers 4 --out <dir>/val_shard0.json

python -m src.quantize.calibration_sensitivity combine \
    --calib artifacts/experiments/day2_calibration_sensitivity_20261120/calibration_sets.json \
    --shards artifacts/experiments/day2_calibration_sensitivity_20261120/kaggle_val/val_shard*.json.gz \
    --out artifacts/experiments/day2_calibration_sensitivity_20261120
```

Kaggle kernels (private, CPU, version that produced the results):
`mehrdadsheikhabbasi/fpga-fault-day2-calib-sens-shard-{0..3}` (shard 0 = v2;
v1 failed on the raw-data path before any computation). Before running, each
kernel checks by SHA-256 that the attached `fpga-fault-src` core modules are the
ones used locally.

## 5. Evidence files

| File | Content |
|---|---|
| `calibration_sets.json` | 12 configs: file lists, scales, windows, time, memory; canonical reproduction flag; validation file list |
| `kaggle_val/val_shard{0..3}.json.gz`, `*.kaggle.log` | raw per-file predictions, saturation counters and timing from Kaggle |
| `val_summary.json` | all metrics, confusion matrices, bootstrap CIs, saturation per position |
| `val_metrics.csv`, `val_saturation.csv`, `scales_vs_canonical.csv` | tables above |
| `val_predictions.npz` | per-window labels and predictions (int8) for all 13 configs |
| `logs/` | calibration and combine console logs |
