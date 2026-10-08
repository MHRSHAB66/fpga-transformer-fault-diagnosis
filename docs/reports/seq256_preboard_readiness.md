# SEQ=256 INT10 — Pre-board readiness checklist

**Date:** 2026-09-30 · **Model:** F_1layer_seq256_pool4 (INT10) ·
**Provisional/generic part:** XC7Z020-CLG400-1 (not a selected board).

Status legend:
- ✅ **Verified** — measured/re-read from artifacts this session.
- 🟡 **Provisional** — complete, but on a provisional/generic part or with caveats.
- ⛔ **Board-pending** — requires the physical board; not done.
- ⚪ **Optional** — future/nice-to-have.

| # | Item | Status | Evidence / note |
|---|---|---|---|
| 1 | Dataset & file-level split | ✅ Verified | MAFAULDA, file-level split; 95,404 test windows |
| 2 | Trained checkpoint | ✅ Verified | `artifacts/experiments/transformer_f1_seq256_pool4/kaggle_v3/seq256_pool4/scratch/best.pt` |
| 3 | Full-test float metrics | ✅ Verified | float aggregate→quantize acc 0.97656, macro-F1 0.95600 |
| 4 | Full-test INT10 metrics (exact ingress) | ✅ Verified | acc **0.97736** (93,244/95,404), macro-F1 **0.95628** — `int10_ingress_local/combined.json` |
| 5 | 4:1 integer aggregation | ✅ Verified | `aggregate_round_div` == HLS `(|s|+2)>>2`; unit-tested |
| 6 | INT10 quantization contract | ✅ Verified | `input_scale=0.028626115354773118`, clamp ±511 |
| 7 | Weight/scale export | ✅ Verified | `hls/f1_int10_seq256/`, `artifacts/quantized/f_seq256_int10/integer_config.json` |
| 8 | Golden vectors | ✅ Verified | 16 ingress vectors `hls/f1_int10_seq256/golden_ingress1024/` |
| 9 | HLS CSim (bit-exact) | ✅ Verified | **16/16 exact** — `transformer_top_csim.log` |
| 10 | HLS C-synthesis | 🟡 Provisional | est. interval 1,490,527 cyc / **14.90527 ms** @100 MHz (estimate only) |
| 11 | DMA integration (PS + AXI-DMA + IP) | 🟡 Provisional | Vivado block design built; `axi_dma_0` + `transformer_top_0`, 32-bit MM2S/S2MM |
| 12 | Post-route timing | 🟡 Provisional | **WNS +0.036 ns, TNS 0, WHS +0.020 ns, THS 0 — met** |
| 13 | Post-route resources | 🟡 Provisional | LUT 31,687 (59.56%), FF 30,889 (29.03%), BRAM 70.5 (50.36%), **DSP 210/220 (95.45%)** |
| 14 | Routing / DRC | 🟡 Provisional | 66,275/66,275 nets routed, 0 errors; DRC 196 warn/adv, 0 critical |
| 15 | Real-time (computational) | 🟡 Provisional | 14.90527 ms compute < 20.48 ms acquisition period at target clock |
| 16 | Bitstream / HWH / XSA | 🟡 Provisional | generated; BIT+HWH same build (`21c7ae29…`, `449b09bf…`) |
| 17 | DSP-margin build | 🟡 Provisional | `-max_dsp 210` reserves 10 DSP; cost +795 LUT/+286 FF/−0.244 ns WNS and **+1,522 cyc (+0.01522 ms, ~+0.1022%)** latency vs 220-DSP (measured, not zero) |
| 18 | PYNQ driver | ✅ Verified (SW) | `pynq/fault_classifier.py`; NumpyBackend 16/16 bit-exact; HardwareBackend uses HWH-verified BD names |
| 19 | PC monitor / UI | ✅ Verified (SW) | `host/fault_monitor.py` + `frame_protocol.py`; file-replay 16 rx / 0 drop / 0 bad |
| 20 | Mock end-to-end validation | ✅ Verified | window→quantize→NumPy HW-ref→logits→class→frame→PC parse |
| 21 | HardwareBackend (mocked vendor pynq) | ✅ Statically/mock validated | 16 mocked tests: instance lookup, buffers (8192,)/(4,) int32, ctrl 0x81, recv-before-send, row-major, argmax; **not** physical DMA |
| 22 | Standalone deployment bundle | ✅ Statically/mock validated | `artifacts/deploy/seq256_z7020_pynq/`; torch-free hardware path, self-locating config/golden, BIT/HWH basenames match, manifest verifies |
| 23 | Board selection | ⛔ Board-pending | XC7Z020 is provisional/generic; final board not chosen |
| 24 | Board constraints (XDC/pins/clock) | ⛔ Board-pending | none applied; PS preset generic |
| 25 | Physical bitstream loading | ⛔ Board-pending | not programmed |
| 26 | Real DMA transfer | ⛔ Board-pending | not run on hardware |
| 27 | UART communication (on-board) | ⛔ Board-pending | protocol tested in software only |
| 28 | Live end-to-end test (physical PYNQ execution) | ⛔ Board-pending | not performed |
| 29 | Live sensor input path (Stage 9) | ⚪ Optional | future work |

## Summary

Consistent evidence distinctions:
- **software accuracy:** measured (item 4);
- **CSim:** bit-exact 16/16 (item 9);
- **HLS latency:** estimate from scheduling (item 10);
- **Vivado post-route timing/resources:** measured implementation reports (items 12–14);
- **generated BIT/HWH/XSA:** complete on provisional/generic XC7Z020 (item 16);
- **standalone board package:** statically/mock validated (items 21–22);
- **physical PYNQ execution / real DMA / UART / live test:** not performed / pending (items 25–28).

- **Software + functional chain: complete and verified** (items 1–9, 18–20).
- **Hardware realization: complete but provisional** on a generic XC7Z020
  (items 10–17) — CSim bit-exact, DMA-integrated post-route timing met,
  bitstream/HWH/XSA generated.
- **Pre-board deployment package complete and statically/mock validated;
  physical-board validation pending** (items 21–22 validated; items 23–28
  board-pending). No claim of on-board operation is made anywhere in this
  repository.

## Key artifacts

- Report: `docs/reports/hls_seq256_int10_results.md`
- Package: `artifacts/results/seq256_dma_system_dsp_margin_z7020/` (+ `MANIFEST.sha256`, `summary.json`)
- Deploy bundle: `artifacts/deploy/seq256_z7020_pynq/`
- Summary JSON: `artifacts/results/hls_seq256_int10_summary.json`
- Cleanup plan: `docs/reports/seq256_repo_cleanup_plan.md`
