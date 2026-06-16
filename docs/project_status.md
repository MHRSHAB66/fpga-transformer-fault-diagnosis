# Project Status

**Project:** Real-Time Machinery Condition Monitoring Using a Transformer Neural Network on FPGA  
**Student:** Mehrdad Sheikh Abbasi, B.Sc., Amirkabir University of Technology  
**Last updated:** 2026-06-16

---

## Stage Status

| Stage | Name | Status |
|---|---|---|
| 0 | Project setup, environment, config | **COMPLETE** |
| 1 | MAFAULDA data discovery, class mapping | **COMPLETE** |
| 2 | Preprocessing, windowing, DataLoaders | **COMPLETE** |
| 3 | MLP baseline (sanity check) | **COMPLETE** |
| 4 | Lightweight Transformer training | **COMPLETE** |
| 5 | Clean evaluation, metrics export, artifacts | **COMPLETE** |
| 6 | Quantization & fixed-point preparation | **TODO — next** |
| 7 | Hardware specification & golden vectors | TODO |
| 8 | FPGA inference core | TODO |
| 9 | Sensor / lab input path | TODO (optional) |
| 10 | FPGA-to-PC communication (UART) | TODO |
| 11 | PC-side display UI | TODO |
| 12 | Final integration, validation, demo | TODO |

---

## Software Phase (Stages 0–5): COMPLETE

The software/model phase is complete as of 2026-06-15.

**Final model:** Lightweight Transformer classifier  
**Final test accuracy:** 98.22% on 95,404 unseen windows from 391 unseen files  
**Fault recall:** 99.75% | False alarm rate: 2.34%  
**Checkpoint:** `artifacts/runs/20260615_223905_fpga_fault_final_t4/checkpoints/transformer_best.pt` (381 KB)

See `docs/final_software_result.md` for full metrics.

---

## What is NOT done yet

- FPGA implementation does not exist.
- No hardware inference has been measured.
- No real-time latency has been measured.
- No UART communication has been implemented.
- No PC display UI exists.
- Quantization (Stage 6) has not started.

Do not claim any of the above in presentations, reports, or demos.

---

## Branch

Current branch: `day11-code-review-and-presentation-prep`

The main branch (`main`) contains the last clean committed state before the final training run.
