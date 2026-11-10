# Paper readiness review (2026-11-10)

Purpose: decide what the B.Sc. system needs, beyond the thesis, to be a credible
paper, using a short literature search and measurements that can be taken while the
Z-Turn board is still available (until about 2026-12-09). Every number below names
its evidence file; estimates are labelled.

## 1. Where the work sits (literature search, 2026-11-10)

| Work | Model / platform | What it reports | Relevance |
|---|---|---|---|
| 1D-ViT bearing diagnosis on FPGA ([synapsesocial](https://synapsesocial.com/papers/69eb0bc7553a5433e34b562e)) | 1D vision transformer, FPGA | ~0.108 ms latency, ~4.59 W, ~98 % accuracy, power vs CPU/GPU | the closest transformer-on-FPGA vibration work |
| BearingPGA-Net, IEEE TIM ([paperswithcode](https://astro.paperswithcode.com/paper/bearingpga-net-a-lightweight-and-deployable)) | CNN, Verilog FPGA | >200× faster than CPU, <0.4 % F1 loss | CNN baseline for FPGA bearing diagnosis |
| Chung et al., Sensors 2023 ([doaj](https://doaj.org/article/4a85930c899d44098f36d64ca468fbba)) | 1D CNN accelerator | 0.28 s response, 96.37 % precision | real-time bearing diagnosis on FPGA |
| Ling et al., IEEE AIoT 2024 ([arXiv 2407.11041](https://arxiv.org/abs/2407.11041v1)); Ling et al. 2025 ([arXiv 2505.17662](https://arxiv.org/abs/2505.17662)) | integer-only tiny transformers, Spartan-7 | measured power and energy per inference (tens to hundreds of µJ) | the integer-only transformer line our core belongs to; sets the energy reporting standard |
| Vieira et al. 2025 ([arXiv 2509.22267](https://arxiv.org/abs/2509.22267v1)) | evaluation methodology | segment-/condition-wise splits inflate accuracy; bearing-wise splits | reviewers will ask how our split controls leakage |
| TSViT ([arXiv 2311.06916](https://arxiv.org/pdf/2311.06916)) and others | transformer diagnosis | Gaussian-noise robustness 0–10 dB (definitions vary) | noise robustness is an expected experiment |
| MAFAULDA rotor-perception card ([huggingface](https://huggingface.co/datasets/AI4Manufacturing/MAFAULDA-rotor-perception)) | dataset note | holds out the highest severity; speed differs per file | precedent for severity hold-out on MAFAULDA |

What is new in this project: a transformer whose **whole path from raw ADC samples to
the class runs inside a small Zynq-7020** (integer z-score, INT10 quantization,
integer softmax/LayerNorm/GELU, argmax), **bit-exact with a Python reference on all
95,404 test windows**, with a **board-side hard deadline verified** (0 misses in
100,000 windows) and the result delivered over UART. Few vibration papers verify
bit-exactness or hard real-time on the physical board.

## 2. Conceptual findings about the current evaluation

1. **Split strength.** The project split is stratified random at file level. All
   391 test files have a training file of the same fault type and severity
   (MAFAULDA has 42 such conditions, ~49 rotation speeds each), at a median
   rotation-speed distance of 1.0 Hz (max 3.7 Hz). The test therefore measures
   generalization to new recordings and speeds of known conditions, not to unseen
   severities. → stricter splits (section 4).
2. **Few normal files.** Only 49 normal files exist; the test has 10. File-level
   bootstrap (2,000 replicates, stratified): accuracy 97.77 % [97.29, 98.26],
   macro-F1 95.67 % [94.34, 96.88], normal recall 87.2 % [79.9, 93.6]
   (`artifacts/results/paper_analysis_20261110/bootstrap_ci.json`). The INT10-input
   and raw-input systems are statistically indistinguishable.
3. **Per-window z-score removes amplitude.** Normal vs. low-severity faults differ
   partly in vibration level, which the normalization discards; the weakest class is
   normal (11.7 % of its windows → misalignment). The on-chip preprocessing already
   computes the per-channel σ, so feeding it to the classifier would cost almost no
   hardware (candidate model change, not done).
4. **Energy is not measured.** Comparable FPGA papers report W and J per inference;
   only a Vivado estimate exists here (section 3).

## 3. Measurements and analyses done on 2026-11-10

| Item | Result | Kind | Evidence |
|---|---|---|---|
| Temporal majority vote over the last k windows (causal, per recording) | k=1 97.77 %; k=5 (102 ms) 99.56 %; k=9 (184 ms) 99.78 %; normal recall 87 → 96 → 98 % | board predictions + analysis | `paper_analysis_20261110/temporal_vote.json` |
| Latency breakdown on the board (R1 / raw-input RT runs) | copy + MM2S 0.83–0.87 ms; core + S2MM + class 14.95 ms (INT10 input) / 15.13 ms (raw input) | MEASURED-PHYSICAL | `hard_rt_*/…/rt_raw.npz` |
| Vivado vectorless power | transformer IP 0.52 W (INT10 input) / 0.63 W (raw input); PS7 1.54 W; on-chip total 2.24 / 2.35 W; confidence "Medium" | ESTIMATE | `paper_analysis_20261110/power_estimate_vivado/` |
| Same model on the board ARM | 216–219 ms per window → FPGA ~14× faster | MEASURED-PHYSICAL | `docs/reports/cpu_baseline_20261109.md` |
| Noise robustness (deployed model, no retraining; C-model = board) | clean 97.77 %; 30 dB 97.77; 20 dB 97.73; 10 dB 96.77; 5 dB 93.16; 0 dB 82.78; normal recall collapses below 10 dB | MEASURED-SOFTWARE (bit-exact) | `artifacts/results/robustness_20261110/summary.json` |
| Dead sensor (channel reads 0) | tachometer 97.77 % (harmless); bearing-1 accelerometers 76–80 %; bearing-2 accelerometers ~90 %; microphone and tach + mic: see summary | MEASURED-SOFTWARE (bit-exact) | same |

## 4. Work in progress (this session)

- Stricter splits retrained with the deployed recipe on the local GPU
  (`tools/generalization_study.py`): unseen middle severity, unseen highest
  severity, unseen 37–47 Hz speed band, plus the project split as control.
- Noise-augmented training, evaluated on the exact raw-domain noisy test windows.
- Temporal vote implemented **in FPGA logic** (`temporal_vote.h`, AXI-Lite
  `vote_k` / `vote_class` / `vote_count`; C-model 41/41 vs the Python reference);
  bitstream build and board validation pending.

## 5. Still needed from the hardware side (board available until ~2026-12-09)

1. Measured power/energy: inline USB or DC power meter (idle, overlay loaded,
   sustained inference; energy per inference = ∫P dt / windows).
2. Board validation of the vote overlay (full test as one stream, change-detection
   delay at recording boundaries).
3. An overnight endurance run of the raw-input core under the guard.
