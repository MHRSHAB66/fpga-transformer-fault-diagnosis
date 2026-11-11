# Noise-augmented weights on the raw-input + vote core (paper study, 2026-11-11)

NOT the deployed model. Same top, preprocessing, vote and system as
`../seq256_dma_vote_z7020_20261110/`; only `weights_int10.h` differs, generated
from `artifacts/quantized/f_seq256_int10_noiseaug/` (checkpoint
`artifacts/results/generalization_20261110/random_noise/best.pt`). Built with
`powershell -File tools/build_raw_core.ps1 -Variant votena`.

| | deployed weights | noise-augmented weights |
|---|---|---|
| HLS C-simulation (own golden set) | 41/41 + votes 41/41 | 41/41 + votes 41/41 |
| Post-route Slice LUTs / registers | 36,911 / 36,349 | 36,788 / 36,358 |
| DSP / BRAM tiles | 210 / 76.5 | 210 / 76.5 |
| WNS / WHS at 100 MHz | +0.135 / +0.007 ns | +0.035 / +0.012 ns |

Deployment bundle: `artifacts/deploy/seq256_vote_noiseaug_z7020_20261111/`
(golden `artifacts/quantized/f_seq256_int10_noiseaug/golden_raw1024_argmax`,
config from the variant export). Board reference logits:
`artifacts/results/noiseaug_int10_20261111/csim_raw_argmax/csim_logits.txt`.
