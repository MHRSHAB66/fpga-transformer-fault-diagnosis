# HLS latency attribution — F_1layer int10 core

Engineering note, weekly report 11. English internal log, not the Persian report.

## Status of evidence

- **Measured:** one number only — top-level latency **74,246,781 cycles** from the
  frozen csynth report
  (`artifacts/results/hls_z7020_q40_bounded_snapshot/transformer_top_csynth.rpt`,
  Vitis HLS 2020.2, `xc7z020clg400-1`, 10 ns target).
- **Not available on this machine:** per-function HLS latency breakdown and any new
  component-probe synthesis. Vitis HLS is **not installed** on the development
  desktop (only a desktop g++ toolchain and Python/NumPy). The per-function
  attribution below is therefore **analytical**, derived from the actual loop
  structure and pragmas in `hls/f1_int10/vitis/transformer_top.cpp`, and
  reconciled against the single measured latency.

Reproduce the model: `python docs/reports/analyze_hls_bottleneck.py`
→ `artifacts/results/hls_attention_bottleneck_analysis.json`.

## Method

Each loop carrying `#pragma HLS PIPELINE` is modelled at II=1 (≈ trip-count
cycles). A loop nested inside a pipelined loop is fully unrolled (parallel
datapath, no extra cycles). Pipeline fill/drain and control handshake are **not**
guessed per loop; they are collected into one reconciled residual.

## Headline finding: cycles are not MACs

The saved workload note is a **MAC count**, in which attention scores and context
are symmetric: 33,554,432 MACs each, 88.58% of all matrix MACs combined. That is
correct but it is **not** the cycle bottleneck, because the two loops are
structured differently in the HLS source:

| Attention step | Loop structure (baseline) | MAC/cycle | Cycles/head | Cycles (×4 heads ×1024 tokens) |
|---|---|---:|---:|---:|
| Scores `Q·Kᵀ` | `j`(SEQ) pipelined, `d`(HD) **unrolled** | 8 | 1024 | 4,194,304 |
| Context `A·V` | `d`(HD) outer, `j`(SEQ) pipelined | 1 | 8192 | **33,554,432** |

Equal MACs, **8× different cycle cost**. Baseline context accumulation alone is
**≈45.2%** of the measured 74.2 M cycles; scores are only **≈5.6%**. The context
loop never unrolls its HD=8 dot-product lanes, so it runs at 1 MAC/cycle while the
score loop runs at 8.

## Full analytical cycle budget

- Ideal II=1 total (model): **50,985,091** cycles.
- Measured total: **74,246,781** cycles.
- Reconciliation residual: **23,261,690** (≈31.3% of measured), attributed to:
  1. the **runtime integer division** in softmax normalisation
     (`qattn = numerator / denom`, variable denominator — a 32-bit divider does
     not pipeline at II=1 on this part);
  2. pipeline fill/drain across the many short per-token inner loops (the baseline
     context loop restarts the `j` pipeline HD times per head);
  3. per-token module handshake (`finish_token` is `INLINE off`).

Largest components by modelled cycle share (of measured total). **These per-row
shares are ANALYTICAL** (the modelled cycle count of each loop divided by the
*measured* 74,246,781-cycle total); they are not directly measured per-function
HLS latencies. In particular the **45.2% context share is an analytical estimate
normalized against the measured total latency**, not a Vitis per-function number.

| Component | Cycles | % of measured |
|---|---:|---:|
| attn_context (baseline) | 33,554,432 | 45.2% |
| kv_proj | 65,536 | 0.09% |
| attn_scores | 4,194,304 | 5.6% |
| attn_rowmax | 4,194,304 | 5.6% |
| attn_exp_denom | 4,194,304 | 5.6% |
| attn_softmax_div (ideal part) | 4,194,304 | 5.6% |
| finish_token | 423,936 | 0.57% |
| input_proj / pe_add / pooling | 32,768 each | 0.04% each |

(The softmax division's *extra* cost beyond II=1 lives in the residual, not in the
row above.)

## Selected optimization

Restructure context accumulation to mirror the score loop: **`j` outer (pipelined),
HD=8 lanes unrolled**, with `v` cyclic-partitioned (factor 8) so the 8 lane reads
land in distinct banks. Implemented in
`hls/f1_int10/vitis/transformer_top_opt.cpp` (isolated; the frozen baseline is
untouched).

**Bit-exactness:** for each output `d`, the accumulator still computes
`Σ_j qattn[j]·v[j·D+off+d]` in ascending `j`. Only the loop-nest order changes;
accumulator width, integer terms, and `requant()` are unchanged. Verified on this
machine by g++ C-simulation: 16/16 golden vectors, output **byte-identical** to
the baseline top (`tmp/gpp_csim_build/`, harness `hls/f1_int10/gpp_csim/`).

**Projected effect (ANALYTICAL, made before synthesis — retained for the record):**

- context: 33,554,432 → 4,194,304 cycles (−29,360,128).
- ideal total: 50,985,091 → 21,624,963 cycles → **2.36× ideal speedup**.
- projected measured latency: **44.9 M (conservative)** to **31.5 M (optimistic)**
  cycles, i.e. **1.65×–2.36×**, throughput **≈2.2–3.2 windows/s** (baseline 1.35).

## MEASURED UPDATE — 2026-09-25 (Vitis HLS 2020.2, xc7z020clg400-1, 10 ns)

The optimization was actually synthesized. Full data:
`artifacts/results/hls_z7020_opt_ctx_measured_summary.json` and the preserved
report `artifacts/results/hls_z7020_opt_ctx_v1_aggressive_inline/`.

- **Vitis C-simulation (ap_int widths): 16/16 exact** — a stronger check than the
  earlier g++ native-width run.
- **MEASURED latency: 36,993,244 cycles (0.370 s), 2.007× faster** than the baseline
  74,246,781 cycles. This lands inside the projected 1.65×–2.36× range, so the
  latency model was sound.
- **MEASURED resources: BRAM 382 (136%), DSP 186 (84%), FF 30,752 (28%),
  LUT 42,350 (79%).** The core **OVERFLOWS Zynq-7020 BRAM** and does **not** fit
  as-is. The projection's implicit "it will fit" assumption was **wrong**.
- Root cause of the BRAM blow-up: the synthesizable variant needs
  `#pragma HLS INLINE` (on `attention0_row`/`attention0_kv` so the `v` partition
  reaches the context loop, and on `linear` so each call keeps a compile-time
  `Din` — without it `transformer_core` latency is unbounded `?`). Inlining
  `linear` unrolls its inner MAC loop, forcing multi-port weight ROMs:
  `finish_token` FFN alone is W_l0_ff3=126 + W_l0_ff0=62 BRAM. The `v` cyclic
  partition itself is cheap (24 BRAM).
- The **no-inline** variant (context reorder + `v` partition only) **hangs 2020.2
  csynth** (stalls at "Starting code transformations", no report;
  `artifacts/results/hls_z7020_opt_ctx_v2_noinline_HANG/`).

Net: **measured 2.0× latency win, but not a viable Z7020 core yet** (BRAM
overflow). Next step: keep the context parallelism while preventing weight-ROM
replication (cap `linear` unroll / bind ROM+multiplier instances, or buffer each
head's V-slice locally), then re-measure fit.

## Honest limit

Even the optimistic projection is far below the **48.83 windows/s** continuous
non-overlapping input rate (20.48 ms/window at 50 kHz). This optimization is
**necessary but not sufficient** for real-time. The next levers, in order, are:
(1) attack the softmax division residual with a hardware-friendly reciprocal
(only if bit-exactness is preserved or the logit deviation is quantified on the
full int10 test set); (2) head-level parallelism (4 heads in parallel);
(3) pipelining the per-token loop across tokens. None of these is claimed as done.

## Discrepancy noted during audit

`artifacts/results/hls_z7020_current_summary.json` stores
`quadratic_attention_share_percent = 88.58329943`; recomputed from its own MAC
counts (67,108,864 / 75,760,192) the exact value is **88.58064140**. Immaterial
(both round to 88.58%) but flagged for correction.
