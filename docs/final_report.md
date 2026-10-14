# Final Project Report

*Real-Time Machinery Condition Monitoring with a Transformer on FPGA*  
Generated 2026-09-09 -- auto-assembled by `src/reporting/generate_final_report.py` from committed artifacts.

> **⚠ Superseded snapshot (2026-09-09).** This auto-generated report describes the
> earlier **SEQ=1024** F_1layer variant (int10 full-test 97.11 %) *before* the
> physical hardware work. The **deployed system is SEQ=256 INT10**, physically
> validated on a MYIR Z-Turn XC7Z020: **full-test accuracy 97.7684 %, macro-F1
> 95.6539 %**; live multiclass 95.0 %. For the canonical deployed-system numbers and
> the live demo flow, see **`docs/demo_runbook.md`** and **`docs/project_status.md`**.

## 1. Model & data

- **Deployed model:** F_1layer (1 encoder layer, d_model=32, 4 heads, ffn=64, 9,492 params)
- **Dataset:** MAFAULDA, 8 channels, window=1024, 4 classes (normal/imbalance/misalignment/bearing_fault)
- **FPGA numeric format:** int10 (activations, weights, attention all 10-bit)

## 2. Software accuracy (full test set)

Measured on 95,404 unseen windows.

| precision | accuracy |
|-----------|----------|
| float (reference) | 97.55% |
| **int10 (FPGA)** | **97.11%** |

**int10 per-class accuracy**

| class | n | accuracy |
|-------|---|----------|
| normal | 2,440 | 99.34% |
| imbalance | 16,348 | 98.46% |
| misalignment | 24,156 | 97.12% |
| bearing_fault | 52,460 | 96.58% |

## 3. Quantization bit-width sweep (F_1layer, 4000-window subset)

| bits | int accuracy | drop vs float | agreement |
|------|--------------|---------------|-----------|
| 16 | 96.23% | +0.17% | 99.25% |
| 14 | 96.17% | +0.22% | 99.70% |
| 12 | 96.15% | +0.25% | 99.67% |
| 10 | 96.38% | +0.02% | 98.88% |
| 8 | 88.67% | +7.72% | 89.78% |

int10 is the smallest near-lossless width; int8 collapses the imbalance class. Hence `ap_int<10>` on hardware.

## 4. Hardware export config

- geometry: seq=1024, in_ch=8, d_model=32, heads=4, layers=1, ffn=64, classes=4
- bit-widths: act=10, weight=10, attn=10; input_scale=0.061840288793037546
- golden self-check: 15/16 classified; bit-exact verify PASS 16/16 (`hls/f1_int10/verify_export.py`)

## 5. Hardware resources & latency (fill after Vitis synthesis)

- target part: TBD -- final FPGA board/device not selected

| metric | value |
|--------|-------|
| LUT | TBD |
| FF | TBD |
| BRAM | TBD |
| DSP | TBD |
| latency_cycles | TBD |
| II | TBD |
| Fmax_MHz | TBD |

See `hls/f1_int10/vitis/SYNTHESIS_GUIDE.md` for the synthesis flow.

## 6. End-to-end pipeline status

| stage | status |
|-------|--------|
| software training | done |
| quantization int10 | done (full-test 97.11%) |
| hls export and verify | done (verify_export PASS 16/16 bit-exact) |
| hls synthesis | TBD (see hls/f1_int10/vitis/SYNTHESIS_GUIDE.md) |
| pc uart receiver and ui | done (host/, tested end-to-end) |
| pynq driver | done (numpy backend verified; hardware backend template) |
| on board uart demo | TBD (needs synthesized bitstream + board) |
