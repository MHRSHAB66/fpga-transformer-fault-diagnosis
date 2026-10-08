# Weekly report 11 — engineering log (HLS latency optimization)

Internal English log. Not the Persian report. Paired with:
`docs/reports/weekly_11_report_outline_fa.md` (proposed Persian outline).

Provisional test part `xc7z020-clg400-1`; **final board not selected** (AGENTS.md).
"Measured" = Vitis HLS 2020.2 csim/csynth.

> **MEASURED UPDATE 2026-09-25:** Vitis HLS 2020.2 was available this session and
> the optimization was actually run. The synthesizable variant is **Vitis CSim
> 16/16 exact (ap_int)** and **MEASURED 2.007× faster** (36,993,244 vs 74,246,781
> cycles), **but it OVERFLOWS Zynq-7020 BRAM (382/280 = 136%)** and does not fit
> as-is. A no-inline variant **hangs** 2020.2 csynth. Full data:
> `artifacts/results/hls_z7020_opt_ctx_measured_summary.json`. Sections below that
> say "projected/to be measured" predate this run and are kept for the record;
> measured values are added, never silently overwritten.

> **MEASURED UPDATE 2026-09-25 (v3 BRAM-fixed) — FITS THE DEVICE.** A follow-up
> variant `hls/f1_int10/vitis/transformer_top_opt_bram.cpp` replaces the generic
> `linear` with a compile-time template `linear_fixed<ROWS,DIN,DOUT>` whose inner
> MAC loop is pipelined (rolled) → weight ROMs stay **single-port, not
> replicated**. **Vitis CSim 16/16 exact; CSynth completed (no hang).** MEASURED
> **44,930,707 cycles = 1.65× faster** than baseline, **and it FITS
> xc7z020clg400-1**: BRAM 108/280 (38.6%), DSP 46/220 (20.9%), FF 23,736 (22.3%),
> LUT 26,857 (50.5%); estimated clock 7.300 ns, slack 0.00 ns. This is the
> recommended candidate. Reports: `artifacts/results/hls_z7020_opt_ctx_v3_bram_fixed/`.

## Measured 3-way comparison (all Vitis HLS 2020.2, xc7z020clg400-1, 10 ns)

| Variant | Latency cyc | Latency ms | Interval | Est.clk (ns) | Slack (ns) | BRAM_18K | DSP | FF | LUT | CSim | Fits? | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|---|
| baseline (frozen) | 74,246,781 | 742.5 | 74,246,782 | 7.297 | n/r | 104 (37%) | 40 | 20,610 | 24,738 | 16/16 | yes | `hls_z7020_q40_bounded_snapshot/` |
| v1 aggressive-inline | 36,993,244 | 369.9 | 36,993,245 | 8.253 | +0.00* | **382 (136%)** | 186 | 30,752 | 42,350 | 16/16 | **NO (BRAM)** | `hls_z7020_opt_ctx_v1_aggressive_inline/` |
| v2 no-inline | — | — | — | — | — | — | — | — | — | 16/16 | **HANG (no report)** | `hls_z7020_opt_ctx_v2_noinline_HANG/` |
| **v3 BRAM-fixed** | **44,930,707** | **449.3** | 44,930,708 | **7.300** | **0.00** | **108 (39%)** | 46 | 23,736 | 26,857 | 16/16 | **YES** | `hls_z7020_opt_ctx_v3_bram_fixed/` |

Speedup vs baseline: v1 = 2.007× (does not fit), **v3 = 1.65× (fits)**. Limiting
resource for v3 is LUT (50.5%). Real-time (48.83 wps) still not met: v3 gap ≈ 22×.
(*v1 top-level slack was reported ~+0.95 ns in its run; value shown here is nominal.)

All numbers above are **measured** by Vitis HLS C-synthesis (estimates, not
post-implementation). Fitting the *core* is not full-system fit; board-readiness
still requires implementation/bitstream + hardware validation.

## Completed

- Frozen baseline audit: all 7 snapshot hashes match the manifest; live
  `hls/f1_int10/vitis/` source is byte-identical to the frozen snapshot.
- Independent recomputation of every derived baseline number (latency seconds,
  throughput, acquisition, real-time gap, fmax, utilizations, MAC totals) — all
  consistent with `hls_z7020_current_summary.json`, except one immaterial
  rounding error (below).
- Analytical latency attribution reconciled to the measured 74.25 M cycles.
- Bit-exact context-accumulation optimization implemented in an isolated file.
- On-machine g++ C-simulation harness for the actual synthesizable tops.
- Report artifacts: bottleneck JSON, optimization summary JSON, figure, this log,
  Persian outline, laptop synthesis TCL.

## Verified

- **Verified by NumPy second-source oracle** (`hls/f1_int10/verify_export.py`):
  16/16 bit-exact logits.
- **Verified by g++ reference testbench** (`hls/f1_int10 && make test`): 16/16.
- **Verified by g++ C-simulation of the synthesizable baseline top**: 16/16.
- **Verified by g++ C-simulation of the optimized top**: 16/16, and output is
  **byte-identical** to the baseline top across all 16 vectors (A/B diff clean).
- **Verified by hash**: baseline provenance intact.

## Measured (from frozen baseline csynth — not re-run)

| Metric | Baseline (measured) |
|---|---:|
| Latency | 74,246,781 cycles ≈ 0.742 s/window |
| Throughput | 1.35 windows/s |
| Estimated clock | 7.297 ns (≈137 MHz) |
| BRAM_18K / LUT / FF / DSP | 104 / 24,738 / 20,610 / 40 |
| Device capacity | 280 / 53,200 / 106,400 / 220 |
| C-simulation | 16/16 exact, 0 errors |

## Failed / inconclusive

- **v1 optimized core does not fit Z7020** (MEASURED): 2.007× faster but BRAM
  382/280 = 136% overflow (weight-ROM replication from inlining `linear`). Not a
  viable core as-is; needs a ROM-sharing/unroll-cap rework.
- **v2 no-inline variant hangs 2020.2 csynth** (MEASURED failure): stalls at
  "Starting code transformations", no report; stopped safely. Evidence:
  `artifacts/results/hls_z7020_opt_ctx_v2_noinline_HANG/`.
- **Softmax runtime-division rework**: identified as ~part of the 31% residual
  but **not implemented** — it risks bit-exactness and the task forbids promoting
  an approximate reciprocal without full-test evidence. Left as a flagged
  candidate, not attempted this week.
- **Reconciliation residual (23.3 M cycles, 31%)** is attributed analytically
  (division II + pipeline fill/drain + handshake) but **not** confirmed per
  function; only a Vitis per-instance report can confirm the split.

## Latency attribution (headline)

Scores and context have **equal MAC counts** (33.55 M each) but **unequal cycle
cost**: the score loop unrolls its HD=8 dot product (8 MAC/cyc) while the baseline
context loop does not (1 MAC/cyc). So baseline context ≈ **45.2%** of measured
cycles vs scores ≈ **5.6%** — an 8× asymmetry. Context is the correct first target.
Details: `docs/reports/hls_attention_bottleneck_analysis.md`.

## Optimization and projection

Restructure context to `j`-outer / HD-unrolled with a `cyclic factor=8` partition
on `v` (`hls/f1_int10/vitis/transformer_top_opt.cpp`). Bit-exact by construction
(same per-`d` ascending-`j` accumulation), confirmed byte-identical under g++.

| Quantity | Baseline (measured) | Optimized (projected) |
|---|---:|---:|
| Context cycles | 33,554,432 | 4,194,304 |
| Ideal II=1 total | 50,985,091 | 21,624,963 |
| Ideal speedup | — | 2.36× |
| Latency (measured→projected) | 74.25 M cyc / 742 ms | 44.9–31.5 M cyc / 449–315 ms |
| Speedup (projected) | — | 1.65×–2.36× |
| Throughput | 1.35 wps | 2.23–3.18 wps (projected) |
| DSP direction | 40 | +up to ~7 (projected; measure) |

## Real-time assessment

Required continuous non-overlapping rate = **48.83 windows/s** (20.48 ms/window
at 50 kHz). Baseline gap ≈ **36.25×**. Even the optimistic projection leaves a
**≈15–22× gap**. Conclusion: the optimization is **necessary but not sufficient**
for real-time; head-level parallelism and the softmax-division residual are the
next levers. **Real-time is NOT claimed.**

## Weekly-11 summary table (for the report)

| Item | Result | Evidence level |
|---|---|---|
| Baseline latency | 74.25 M cyc, 0.742 s, 1.35 wps | measured (csynth) |
| Baseline resources | 104 BRAM / 24.7k LUT / 20.6k FF / 40 DSP | measured (csynth) |
| Baseline correctness | 16/16 exact | measured (csim) + g++ csim |
| Bottleneck | context accumulation ≈45% cycles | analytical |
| Optimization functional | 16/16 exact (ap_int) + byte-identical to baseline | **measured (Vitis csim)** + g++ |
| Optimization latency (pre-synth) | 1.65–2.36× faster | projected (analytical) |
| **Optimization latency (actual)** | **2.007× faster (37.0 M cyc, 0.370 s)** | **measured (Vitis csynth)** |
| **Optimization resources (actual)** | **BRAM 382/280=136% OVERFLOW, DSP 186, FF 30.8k, LUT 42.4k** | **measured (Vitis csynth)** |
| **Optimization fit on Z7020** | **does NOT fit (BRAM overflow)** | **measured** |
| Real-time (continuous) | not achieved (gap 36.25× → 18.06× at 2.0×) | measured |

## Exact commands

```bash
# audit
sha256sum artifacts/results/hls_z7020_q40_bounded_snapshot/{transformer_top.cpp,ap_types.h,weights_int10.h,testbench_top.cpp,run_verify_and_synth.tcl} \
          artifacts/results/hls_z7020_q40_bounded_snapshot/golden/{golden_in.txt,golden_expected.txt}
python hls/f1_int10/verify_export.py
cd hls/f1_int10 && make test && cd ../..

# on-machine C-sim of the synthesizable tops (needs desktop g++ only)
g++ -O2 -std=c++17 -Wno-unknown-pragmas -Ihls/f1_int10/gpp_csim/stubs -Ihls/f1_int10/vitis \
    hls/f1_int10/vitis/transformer_top.cpp     hls/f1_int10/vitis/testbench_top.cpp -o tmp/tb_top_baseline
g++ -O2 -std=c++17 -Wno-unknown-pragmas -Ihls/f1_int10/gpp_csim/stubs -Ihls/f1_int10/vitis \
    hls/f1_int10/vitis/transformer_top_opt.cpp hls/f1_int10/vitis/testbench_top.cpp -o tmp/tb_top_opt
./tmp/tb_top_baseline hls/f1_int10/golden
./tmp/tb_top_opt      hls/f1_int10/golden

# analysis + figure
python docs/reports/analyze_hls_bottleneck.py
python docs/reports/plot_hls_bottleneck.py

# MEASURED optimized csim+csynth (Vitis HLS 2020.2) — RUN THIS SESSION 2026-09-25.
# Run from a scratch dir so the project stays out of git; -reset guarantees no
# stale results. Produces csim (ap_int 16/16) then csynth report.
mkdir -p tmp/hls_opt_run && cd tmp/hls_opt_run
"G:/Xilinx/Vitis_HLS/2020.2/bin/vitis_hls.bat" -f \
  "G:/aut/project/fpga_fault_diagnosis/hls/f1_int10/vitis/run_top_opt_verify_and_synth.tcl"
# report: tmp/hls_opt_run/opt_ctx_q40_z7020/sol1/syn/report/transformer_top_csynth.rpt
```

## Files created / changed this week

Created:
- `hls/f1_int10/vitis/transformer_top_opt.cpp` — isolated bit-exact optimized top
- `hls/f1_int10/vitis/run_top_opt_verify_and_synth.tcl` — laptop csim+csynth flow
- `hls/f1_int10/gpp_csim/stubs/hls_stream.h`, `.../ap_axi_sdata.h`, `.../README.md`
- `docs/reports/analyze_hls_bottleneck.py` — analytical cycle model generator
- `docs/reports/plot_hls_bottleneck.py` — optimization figure generator
- `docs/reports/hls_attention_bottleneck_analysis.md` — attribution note
- `docs/reports/weekly_11_engineering_log.md` — this file
- `docs/reports/weekly_11_report_outline_fa.md` — Persian outline
- `artifacts/results/hls_attention_bottleneck_analysis.json`
- `artifacts/results/weekly11_optimization_summary.json`
- `artifacts/results/hls_bottleneck_optimization.png`
- **MEASURED (2026-09-25):** `artifacts/results/hls_z7020_opt_ctx_measured_summary.json`
- **MEASURED (2026-09-25):** `artifacts/results/hls_z7020_opt_ctx_v1_aggressive_inline/` (csynth reports + source + sha256)
- **MEASURED (2026-09-25):** `artifacts/results/hls_z7020_opt_ctx_v2_noinline_HANG/` (hang evidence)
- `artifacts/results/hls_measured_opt_vs_baseline.png` (measured figure)

Pre-existing uncommitted (NOT touched by this work): `src/train_transformer_baseline.py`
(+ prior weekly-11 artifacts: `hls_z7020_current_summary.json`,
`weekly11_hls_progress.png`, `plot_weekly11_hls_progress.py`, `weekly_11_evidence_notes.md`).

## Discrepancy log

- `hls_z7020_current_summary.json:quadratic_attention_share_percent` = 88.58329943;
  recomputed exact = **88.58064140**. Immaterial (both round to 88.58%). Flagged,
  not silently edited; corrected value recorded in the bottleneck JSON.

## Remaining blockers (ordered)

1. ~~Optimized csynth measurement~~ **DONE 2026-09-25** (measured; see above).
2. **Make the optimized core FIT Z7020** — v1 overflows BRAM (136%); rework weight
   access to stop ROM replication (cap `linear` unroll / bind ROM+mult instances,
   or buffer each head's V-slice locally to avoid the v2 pointer-partition hang),
   then re-measure. This is the immediate next step.
3. **C/RTL co-simulation** — only meaningful once a *fitting* candidate exists;
   estimate wall time for SEQ=1024 before launching (may need reduced vector count).
4. **IP packaging / bitstream** — after a stable, fitting synthesized candidate.
5. **PYNQ execution + UART + PC-side UI** — Stages 9–11, unchanged.
