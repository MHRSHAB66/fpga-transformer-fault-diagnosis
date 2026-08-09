# Tuning - all 7 configs (validation set)

_Generated 2026-08-01 19:03._

| Rank | Config | Params | Val acc | Macro F2 | Fault recall | False alarm | FPGA |
|-----:|--------|-------:|--------:|---------:|-------------:|------------:|------|
| 1 | D_d64_slow | 69,860 | 99.66% | 0.9965 | 99.96% | 0.08% | good |
| 2 | C_d48 | 39,724 | 99.27% | 0.9898 | 99.82% | 0.98% | ~limited |
| 3 | E_3layers | 26,580 | 98.73% | 0.9852 | 99.77% | 0.57% | good |
| 4 | A_baseline | 18,036 | 98.19% | 0.9823 | 99.70% | 0.25% | excellent |
| 5 | F_1layer | 9,492 | 98.28% | 0.9737 | 99.30% | 1.07% | excellent |
| 6 | B_wide_ffn | 26,356 | 97.46% | 0.9679 | 99.28% | 0.41% | good |
| 7 | P_prior_like | 1,582 | 89.26% | 0.8662 | 96.65% | 6.56% | excellent |

Findings: bigger model -> higher accuracy (D best: val 99.66%, test 99.30%).
A_baseline/F_1layer most FPGA-friendly. P_prior_like (paper-inspired, width 8, 1.6k params)
underperforms badly (val 89.26%, test 89.26%, false-alarm 6.5%) and the kernel's own rule
says KEEP ORIGINAL BASELINE -> justifies our 8->32 input projection.
