# HLS throughput study — approaching real-time on Zynq-7020 (provisional)

Measured, Vitis HLS 2020.2, `xc7z020clg400-1`, 10 ns, `-DHLS_USE_AP_INT`
(`ap_int<10>`/`<32>`/`<20>`). Every variant preserves the approved model
(SEQ=1024, D=32, NHEAD=4, DFF=64, full O(SEQ²) softmax attention), weights,
int10/Q40 scales, LUTs, class IDs, tensor order, AXI interface and golden vectors,
and is **bit-exact** (native g++ A/B byte-identical; Vitis CSim 16/16).

**Real-time requirement:** interval ≤ **2,048,000 cycles** (20.48 ms @ 10 ns, ≈48.83 windows/s).
`xc7z020clg400-1` is a **provisional** measurement target; all figures are
C-synthesis **estimates**, core only (no DMA/PS/bitstream). The top is `ap_ctrl_hs`,
so **interval ≈ latency** (no window overlap).

## Stage 0 — measured bottleneck (v3)

| Item | Cycles | % of total | II | Limit |
|---|---:|---:|---:|---|
| Query loop (LOOP_273) | 42,004,480 | 93.5% | — | not pipelined → interval=latency |
| ↳ scores / head | 4,107 | 37.4% (all heads) | **4** | **K read ports (200-885)** |
| ↳ rowmax / head | 1,023 | 9.3% | 1 | — |
| ↳ exp+num / head | 1,026 | 9.4% | 1 | — |
| ↳ norm+div / head | 1,056 | 9.6% | 1 | — |
| ↳ context / head | 1,027 | 9.4% | 1 | — |
| finish_token / query | 5,889 | 13.4% | — | — |
| K/V projection serial prefix | ~2,883,600 | 6.4% | — | runs before any query |

The **score loop II=4 (K-port limited)** is the #1 within-attention target.

## Stage 0 — theoretical lower bounds (ANALYTICAL projections, not measured)

| Scenario | Interval | Gap | Real-time? |
|---|---:|---:|:--:|
| S1 score II=1, heads sequential | 32,184,969 | 15.7× | no |
| S2 +2 heads parallel | 21,625,481 | 10.6× | no |
| S3 +4 heads parallel | 16,345,737 | 8.0× | no |
| S4 +query DATAFLOW (finish not split) | 8,913,936 | 4.4× | no |
| S5 +finish_token split (FF rolled) | 4,990,992 | 2.4× | no |
| S6 +FF unroll ×2 | 3,964,944 | 1.9× | **no** |
| mem-bw floor (32-way, K/V pre-available) | ~1,048,576 | 0.5× | (not realizable) |

**Gate result:** even the full unchanged-architecture stack (score II=1 + 4-head
parallel + query DATAFLOW + finish split + FF unroll) floors at **~3.9M cycles
(gap ~1.9×)** — still **above** 2,048,000 — because the **serial K/V-projection
prefix (~2.88M)** alone nearly exhausts the budget. The ~1.05M memory-bandwidth
floor is *not* realizable while K/V generation stays a serial rolled prefix.
**Reaching real-time additionally requires accelerating the K/V (and FFN)
projections** — a much larger, higher-risk redesign. This gate was computed
*before* running the expensive parallelism/dataflow experiments.

## Measured results (all bit-exact, CSim 16/16)

| Variant | Interval cyc | ms | wps | RT gap | Est.clk | Slack | BRAM | DSP | FF | LUT | Fit | RT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|
| baseline | 74,246,782 | 742 | 1.35 | 36.3× | 7.297 | — | 104 | 40 | 20,610 | 24,738 | yes | no |
| v1 fast | 36,993,245 | 370 | 2.70 | 18.1× | 8.253 | — | **382** | 186 | 30,752 | 42,350 | **no** | no |
| v3 balanced | 44,930,708 | 449 | 2.23 | 21.9× | 7.300 | 0.00 | 108 | 46 | 23,736 | 26,857 | yes | no |
| **T1 K-banked** | **32,835,209** | 328 | 3.05 | **16.0×** | 8.468 | +1.17 | 112 | 45 | 22,710 | 25,991 | yes | no |
| **T2 +req-share** | 32,863,881 | 329 | 3.04 | 16.0× | 8.468 | +1.17 | 112 | 45 | **18,829** | **22,150** | yes | no |

## Variants attempted (measured)

- **T1 — K-banking (`transformer_top_rt_t1_kbank.cpp`).** Change: `k` cyclic-partitioned
  factor 8 + written by an in-scope loop (mirroring how v3 banks `v`), so the score
  dot-product reads HD=8 k-lanes/cycle. **Measured: score loop II 4→1** (the
  `200-885` K-port warning disappeared); latency **44.93M → 32.84M (1.37× faster)**;
  BRAM +4 (k banks); fits 7020; slack +1.17 ns. Bit-exact.
- **T2 — shared context-output requant (`transformer_top_rt_t2_reqshare.cpp`).** Change:
  rolled the HD context-output requant loop → one shared requant datapath. **Measured:
  LUT 25,991→22,150 (−3,841), FF 22,710→18,829 (−3,881)** at **+28,672 cycles**
  (negligible). Frees resources for future parallelism. Bit-exact.

## Analyzed but NOT implemented this session (justified stop)

Per stop-condition "stop if the theoretical best continuation still cannot reach the
target," and because the gate shows the unchanged-architecture floor (~3.9M) is above
2,048,000:
- **T3 head parallelism** (2/4 heads): needs K/V partition factor 16/32 + head-loop
  unroll; projected S2/S3 = 21.6M/16.3M (gap 10.6×/8.0×) — still far from real-time.
- **Stage 4 query DATAFLOW**: the only route below ~4M, but even the full stack floors
  at ~3.9M (> target) without K/V-projection acceleration, and it carries a high risk
  of the Vitis 2020.2 code-transformation stall previously observed with
  partitioned-array/pointer boundaries.
- **Stage 5 key-lane parallelism**: not warranted before dataflow; risks denominator
  ordering / exactness.

## Selection & real-time verdict

Fastest fitting bit-exact implementation: **T1/T2** at ~32.8M-cycle interval (T2
preferred for its LUT/FF headroom at equal latency). **Real-time was NOT achieved.**
Remaining gap: **32,863,881 / 2,048,000 = 16.0×** (T2).

**Why:** interval = latency (no cross-query overlap); attention is O(SEQ²)=1024², and
the K/V-projection serial prefix (~2.88M) alone exceeds the 2.048M budget.
Within-query optimizations (T1/T2) removed the score-port bottleneck but cannot make
interval ≪ latency.

**Smallest next justified change:** query-loop **DATAFLOW** (split attention into
bounded `hls::stream` stages, pipelined across the 1024 independent queries) +
**4-head parallelism** + **K/V (and FFN) projection acceleration**. Even then the
analytical floor is ~2–4M (marginal); if 2,048,000 must be met, a scope change
(overlap window acquisition with compute, or accept a larger interval) may be
required. This is a dedicated, high-risk follow-up on Vitis 2020.2.

## Timing & board-readiness

Estimated (C-synth) only: target 10 ns, estimated clock 8.468 ns (T1/T2), HLS slack
**+1.17 ns**. **Post-implementation timing is unresolved; not board-ready** (no PAR,
bitstream, DMA/PS integration, or on-hardware validation). Final board not selected.

## Regressions (all pass)
Vitis CSim ap_int 16/16 (every variant); native g++ A/B byte-identical; NumPy oracle
16/16; `unittest` 30/30 OK; `git diff --check` clean except the pre-existing user
change in `src/train_transformer_baseline.py` (untouched). Baseline
`transformer_top.cpp` and v3 `transformer_top_opt_bram.cpp` unchanged.

## Artifacts
`artifacts/results/hls_rt_t1_kbank/`, `artifacts/results/hls_rt_t2_reqshare/`
(reports, source, TCL, SHA-256, CSim/timing), `hls_throughput_study_summary.json`,
figure `hls_throughput_study.png`. Prior baseline/v1/v3/7010/7020 evidence preserved
and not overwritten.
