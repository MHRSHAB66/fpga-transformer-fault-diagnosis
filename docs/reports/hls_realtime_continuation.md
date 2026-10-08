# HLS real-time campaign continuation

Target used for this comparison: `xc7z020clg400-1`, 10 ns. This is an
experimental comparison target, not a final board selection. All numbers below
come from completed Vitis HLS 2020.2 C-synthesis reports and all candidates pass
the 16-vector `ap_int` CSim exactly.

| Candidate | Interval (cycles) | Time at 10 ns | BRAM_18K | DSP | FF | LUT | RT gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| T2 starting point | 32,863,881 | 328.64 ms | 112 | 45 | 18,829 | 22,150 | 16.05x |
| Corrected DATAFLOW | 26,811,336 | 268.11 ms | 131 | 44 | 19,171 | 21,796 | 13.09x |
| DATAFLOW + nominal 2-head unroll | 26,809,288 | 268.09 ms | 147 | 59 | 23,855 | 26,210 | 13.09x |
| K/V projection factor 2 | 31,815,305 | 318.15 ms | 112 | 47 | 18,903 | 22,221 | 15.53x |
| **K/V factor 2 + DATAFLOW** | **25,762,760** | **257.63 ms** | **131** | **46** | **19,245** | **21,894** | **12.58x** |
| **Explicit 2-head processes** | **9,835,467** | **98.35 ms** | **207** | **91** | **33,707** | **36,343** | **4.80x** |
| Explicit 4-head region | 9,835,467 | 98.35 ms | 207 | 91 | 33,704 | 36,329 | 4.80x |
| **Attention x2 + FFN x4** | **7,212,718** | **72.13 ms** | **216** | **160** | **47,046** | **50,197** | **3.52x** |

## Findings

- The original DATAFLOW form failed quickly because `x_in` had two readers.
  Forwarding the query row created a legal load-attention-finish-store chain.
- Corrected DATAFLOW is the largest measured win in this continuation.
- The head-loop `UNROLL factor=2` candidate improves the interval by only 2,048
  cycles while adding 16 BRAM, 15 DSP, 4,684 FF, and 4,414 LUT. It is rejected.
- Factor-2 K/V reductions shorten the serial prefix by about 1.05 million cycles.
  Combining them with DATAFLOW gives the best measured interval, 25,762,760.
- The combined candidate still spends 23,459,529 cycles in the query dataflow
  parent loop, with a per-query dataflow interval of about 22,904 cycles.
  Attention therefore remains the limiting stage. Accelerating only the FFN
  cannot improve the current top-level interval because it is already overlapped
  behind the slower attention stage.

## Verdict and next justified change

Real-time is not achieved: 25,762,760 / 2,048,000 = 12.58x over the acquisition
budget. The next experiment must explicitly create concurrent per-head attention
processes or parallelize the key/token traversal inside the attention stage;
another loop-unroll pragma on the existing shared process is not justified by
the measured two-head result.

These are C-synthesis estimates, not post-implementation or board results.

## Explicit head-parallel continuation

The earlier head-loop unroll did not create useful concurrency. The corrected
experiment split K/V into private per-head memories and instantiated independent
head functions inside DATAFLOW regions. Vitis reports four separate `df_head`
modules and the interval falls from 25,762,760 to 9,835,467 cycles (2.62x).

Putting all four already-instantiated heads into one DATAFLOW region does not
reduce the top interval further. The measured query pipeline is now limited by
two comparable stages: `df_attn` at 7,333 cycles and `finish_token` at 5,761
cycles per query. Reaching 2,048,000 cycles therefore requires accelerating both
attention token traversal and the residual/layernorm/FFN path. Head replication
alone is no longer a justified next step.

## Attention-x2 and FFN-x4 continuation

The next isolated candidate processes two attention tokens per loop iteration
and uses four parallel MAC accumulators in each FFN projection. Both native C++
simulation and Vitis `ap_int` CSim match all 16 golden vectors exactly.

Measured C-synthesis results:

- top interval: 7,212,718 cycles (72.13 ms), 1.36x faster than the explicit
  head-parallel candidate;
- `df_attn`: 4,773 cycles per query, down from 7,333;
- `finish_token`: 4,452 cycles per query, down from 5,761;
- resources: 216/280 BRAM_18K, 160/220 DSP, 47,046/106,400 FF, and
  50,197/53,200 LUT;
- estimated clock: 7.299 ns against a 10 ns target.

The candidate fits the experimental 7020 C-synthesis budget, but only 3,003 LUT
remain. It is not real-time: 7,212,718 / 2,048,000 = 3.52x. The query pipeline
still costs 4,893,100 cycles, while the serial prefix and remaining work account
for about 2,311,415 cycles. Because that non-query portion already exceeds the
real-time budget, further attention/FFN unrolling alone cannot meet the target;
the K/V projection prefix must also be redesigned or overlapped. These remain
C-synthesis estimates, not place-and-route or board measurements.

## Prefix (K/V projection) continuation — weekly 12

Starting point: `rt_attn2_ff4`, interval 7,212,718 cycles (72.13 ms), gap 3.52x,
LUT 50,197/53,200 (94%). Measured non-query prefix = 2,311,415 cycles, broken
down (measured) as: K projection 983,040 + V projection 983,040 + input
projection 262,154 + (pe/pool/classifier/misc) ~83,181. The two K/V projections
are 85% of the prefix. The query dataflow region is 4,893,100 cycles.

Lower bound (measured floor): the prefix and the query dataflow region run in
series, so total interval >= 4,893,100 cycles even with a zero-cycle prefix =
2.39x the 2,048,000-cycle budget. Reducing the query region needs more attention
parallelism, which the ~94% LUT budget cannot afford. Real-time is therefore not
achievable for this unchanged bit-exact architecture on the 7020 C-synthesis
budget. Prefix acceleration can only bring the design materially closer.

| Candidate | K/V prefix | Interval | Time@10ns | BRAM | DSP | FF | LUT | RT gap | Fit | CSim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|
| rt_attn2_ff4 (start) | 1,966,080 | 7,212,718 | 72.13 ms | 216 | 160 | 47,046 | 50,197 | 3.52x | yes | 16/16 |
| prefix_opt1 factor-4 K/V | 2,031,616 | 7,278,254 | 72.78 ms | 216 | 164 | 47,330 | 50,467 | 3.55x | yes | 16/16 |
| **prefix_opt2 fused K+V** | **983,040** | **6,229,677** | **62.30 ms** | **216** | **160** | **46,839** | **49,889** | **3.04x** | **yes** | **16/16** |

Findings (measured):

- **prefix_opt1 (factor-4 K/V) is REJECTED.** Widening each reduction from 2 to 4
  accumulators made it *slower* (K/V loops 983,040 -> 1,015,808 each, interval
  +65,536) and used more LUT. The K/V projection is fill-limited: a wider add-tree
  deepens the pipeline, so the per-output fill grows (14 -> 23 cycles) and more
  than cancels the shorter inner trip. More MAC lanes do not help here.
- **prefix_opt2 (fused K+V) is the new fastest fitting candidate.** Both
  projections read the same input row, so fusing them into one (r,o) loop pays the
  per-output pipeline fill once instead of twice: the K/V prefix halves from
  1,966,080 to 983,040 cycles. Interval falls to 6,229,677 (1.158x), and LUT is
  slightly lower (49,889 vs 50,197) because there is one loop nest instead of two.
  Bit-exact (independent K/V reductions, same order/requant). It fits the 7020
  budget but is not real-time: 6,229,677 / 2,048,000 = 3.04x.

Verdict: real-time is not achieved. The measured hard floor is the 4,893,100-cycle
query dataflow region (2.39x the budget), which cannot be reduced without more
attention parallelism that the ~94% LUT budget does not allow. The next justified
prefix step is to run K and V projection concurrently (analytical estimate ~983k
more saved -> total ~5.2M, gap ~2.5x), but that still cannot reach real-time and
would require duplicating/forwarding the input row for the DATAFLOW single-reader
rule. These remain C-synthesis estimates, not place-and-route or board results.

## K/V concurrent DATAFLOW experiment — weekly 12 (REJECTED)

Hypothesis: run the K and V projections concurrently with HLS DATAFLOW to shorten
the serial prefix of prefix_opt2 (fused K+V = 983,040 cycles).

Analytical estimate (before synthesis): prefix_opt2 already fuses K and V into one
II=1 loop that computes both in 983,040 cycles, so a concurrent split can at best
match ~983,040 and was expected to give no top-level improvement while adding an
input-duplication buffer. Labelled analytical, not measured.

Implementation (`transformer_top_rt_kv_dataflow.cpp`): `attention0_kv` became a
DATAFLOW region `df_split_x || df_kproj || df_vproj`; `x` is duplicated into private
`xk`/`xv` so each projection is the sole reader of its input (dataflow single-reader
rule). Bit-exact: native g++ matches baseline and prefix_opt2 (16/16); Vitis
`ap_int` CSim 16/16.

Measured (Vitis HLS 2020.2, xc7z020clg400-1, 10 ns):

| Metric | prefix_opt2 | kv_dataflow |
|---|---:|---:|
| K/V prefix / region | 983,040 | 983,044 (df_kproj 950,273 ∥ df_vproj 950,273) |
| Interval (cycles) | 6,229,677 | 6,229,681 |
| Time @ 10 ns | 62.30 ms | 62.30 ms |
| BRAM_18K | 216 (77%) | **296 (105% — OVERFLOW)** |
| DSP | 160 | 160 |
| FF | 46,839 | 47,093 |
| LUT | 49,889 | 50,752 |
| RT gap | 3.04x | 3.04x |
| Fit | yes | **no** |
| CSim | 16/16 | 16/16 |

Findings (measured): DATAFLOW concurrency **was** actually inferred — `df_kproj` and
`df_vproj` run in parallel (each 950,273 cycles). But the **top-level interval did
not improve** (+4 cycles) because fusion already realized that concurrency, and the
**BRAM overflows the device (296/280 = 105%)** because of the `xk`/`xv` input
duplication required by the dataflow single-reader rule. **REJECTED** on two stop
conditions (no interval improvement; resource over capacity).

Verdict: `prefix_opt2` remains the fastest fitting bit-exact candidate
(6,229,677 cycles, gap 3.04x). Real-time is not achieved. The measured floor of the
current implementation is the 4,893,100-cycle query dataflow region (this is the
floor of this implementation, not a proven absolute lower bound); reducing it needs
more attention parallelism that the ~94-95% LUT budget cannot afford. Per the plan,
the campaign stops here and a decision is requested before any model/architecture
change. These are C-synthesis estimates, not place-and-route or board results.

## Query-attention analysis and real-time verdict — weekly 12 (Phases 0-2)

Phase 0 (MEASURED cycle model of the 4,893,100-cycle query region, from the
attn2_ff4 df_head/df_attn/finish_token reports): per head the softmax runs as five
SEQUENTIAL passes over the key dimension (score 524, row-max 511, exp 514,
normalize/divide 544, context 515; output 13) = 2,638 cycles; the four heads run
CONCURRENTLY (df_head_all = 2,638); attn2 already reads 2 keys/cycle (trip 512).
Per-query dataflow interval = 4,774 = max(df_attn 4,773, finish_token 4,452); these
already overlap across queries. So the query region is five serial key-length passes
per head, plus a per-token finish (FFN + 2 layernorms) of 4,452.

Phase 1/2 (lower bound, ANALYTICAL): total budget 2,048,000; prefix_opt2 = prefix
1,328,374 + query 4,893,100. Fitting requires the query region to shrink 2.39x even
with a zero prefix (6.8x if the prefix is unchanged). The only mechanism is key-lane
parallelism P>=5..8 over the key dimension plus a >=2.5x finish/FFN speedup. That
roughly doubles-to-quadruples the attention datapath, but prefix_opt2 is already at
94% LUT (3,311 free); P>=4 alone exceeds the 53,200 LUT device limit (DSP 160/220
and FF have room; LUT is binding). No unchanged-model query variant passes the
synthesis gate (bit-exact pass-fusion floors ~4.6M > target; key-lane 8x is
LUT-infeasible), so none was synthesized.

Verdict (Phase 2): real-time (<=2,048,000 cycles at 10 ns) is NOT achievable with
the unchanged four-class int10 model on xc7z020clg400-1. The query attention region
is LUT-bound. This is supported by measured evidence: prefix_opt2 at 94% LUT with a
4.89M query region, explicit 4-head giving no gain over 2-head, attn2+ff4 already at
94% LUT, and kv_dataflow overflowing BRAM (296/280).

Phase 3 (AUTHORIZED next branch, ANALYTICAL): token aggregation mapping the 1024x8
window to SEQ=256 tokens (4:1), preserving all eight channels and the 20.48 ms
window, then retraining the same F_1layer transformer and re-quantizing to int10.
Because attention is O(SEQ^2), the query region drops toward a finish-limited
256 x 4,452 ~= 1.14M, and the prefix to ~0.33M, for a total ~1.47M cycles
(analytical, ~1.4x under budget) -> real-time plausibly achievable. This requires a
retrain + re-quantization + regenerated golden vectors + measured accuracy on the
preserved file-level split + a new HLS top + synthesis. GPU, processed data, and the
train/quantize/export scripts are available locally, so it is feasible; it is a
dedicated multi-step cycle and is not yet executed. All Phase-3 numbers here are
analytical estimates, not measurements.

## Softmax pass-reduction feasibility gate — weekly 12 (hardware-only, no retraining)

Goal: reduce the attention softmax/context key-length passes (score, row-max, exp,
normalize, context) without changing/retraining the model.

Phase 0 numerical contract (MEASURED from prefix_opt2 df_head): scores are
`requant(q·k, M_scores)` clamped to int10; row-max exact; `num[j]=exp_l0[sc[j]-max]`
(int20 LUT) with `denom += num[j]`; `qattn[j]=(num[j]*511+denom/2)/denom`
(round-half-up, int10); `ctx[d]=requant(Σ qattn[j]*v[j][d], M_ctx)`. Five sequential
key-length passes (attn2 = 2 keys/cycle, trip 512).

Fusion classification: fusing score+max and fusing normalize+context is BIT-EXACT
(every stored value and accumulation order is preserved) → 16/16 exact, no accuracy
change, 5→3 passes. Moving ÷denom to the end (two-pass) or an online/streaming
softmax removes the per-key int10 rounding of qattn → APPROXIMATE (would need
accuracy evaluation and new candidate-specific golden vectors).

Phase 1 feasibility gate (MEASURED stage balance + ANALYTICAL bound): the per-query
dataflow interval is `max(df_attn 4,773, finish_token 4,452)` (both measured; they
already overlap across queries). Reducing softmax passes only shrinks df_attn; once
it drops below 4,452, **finish_token (4,452) is the hard floor**. Hence every
formulation converges to the same total:

| Formulation | passes | per-query II | total interval | vs prefix_opt2 | RT gap | bit-exact |
|---|---:|---:|---:|---:|---:|:--:|
| current (prefix_opt2) | 5 | 4,774 | 6,229,677 | 1.00x | 3.04x | yes |
| bit-exact 3-pass | 3 | 4,452 (finish) | ~5,895,425 | 1.057x | 2.88x | yes |
| approximate 2-pass | 2 | 4,452 (finish) | ~5,895,425 | 1.057x | 2.88x | no |
| online softmax | 1 | 4,452 (finish) | ~5,895,425 | 1.057x | 2.88x | no |

Verdict: **rejected before synthesis.** All softmax-pass reductions converge to the
finish_token-limited floor ~5,895,425 cycles — only 1.057x (5.4%) faster than
prefix_opt2 and still 2.88x from real-time. Per the Phase-1 gate ("do not synthesize
a design whose rigorous lower bound cannot materially improve 6,229,677"), no
candidate is synthesized. The bit-exact 3-pass is arithmetically free but not worth
it; the approximate variants give the SAME interval while costing accuracy, so they
are strictly worse. **finish_token/FFN is the co-bottleneck.**

Because finish_token runs once per token, and attention is O(SEQ^2), only reducing
the number of tokens attacks BOTH limiters at once. The next justified experiment is
therefore the isolated SEQ=256 token-aggregation/retraining branch (NOT started;
requires retrain + re-quantize + new goldens + accuracy). Real-time was not achieved
here. These are C-synthesis estimates (finish_token/df_attn measured; fused totals
analytical).
