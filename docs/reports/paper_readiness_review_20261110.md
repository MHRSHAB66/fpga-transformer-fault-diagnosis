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
| Dead sensor (channel reads 0) | tachometer 97.77 % (harmless); bearing-1 accelerometers 76–80 %; bearing-2 accelerometers ~90 %; microphone 84.2 %; tachometer + microphone 78.1 % (figure `docs/figures/paper/dead_sensor`) | MEASURED-SOFTWARE (bit-exact) | same |

## 4. Generalization study: stricter file-level splits (float, retrained)

Same architecture (F_1layer SEQ=256) and the deployed training recipe, retrained
on the local GPU for every split (`tools/generalization_study.py`; evidence
`artifacts/results/generalization_20261110/<split>/result.json`). Severity
splits hold out one level of every fault family (test) and another one
(validation); normal files have no severity, so a speed band of the normal class
is held out instead.

| Split | Files train / val / test | Test windows | Accuracy % | Macro-F1 % | Macro AUROC | Recall % normal / imbalance / misalignment / bearing |
|---|---|---|---|---|---|---|
| random — project split (control) | 1365 / 195 / 391 | 95,404 | 97.47 | 95.01 | 0.9990 | 95.8 / 97.5 / 98.3 / 97.2 |
| speed_mid — unseen 37–47 Hz band | 1339 / 212 / 400 | 97,600 | 97.88 | 95.28 | 0.9991 | 95.7 / 97.4 / 99.5 / 97.3 |
| sev_mid — unseen middle severity | 1084 / 439 / 428 | 104,432 | 95.74 | 90.60 | 0.9967 | 79.2 / 99.7 / 91.4 / 97.2 |
| sev_max — unseen highest severity, normal ≥ 52 Hz | 1158 / 420 / 373 | 91,012 | 74.26 | 55.10 | 0.9175 | 1.4 / 83.6 / 96.2 / 65.7 |
| sev_max_nmid — unseen highest severity, normal middle band | 1157 / 421 / 373 | 91,012 | 74.09 | 72.42 | 0.9271 | 79.1 / 77.9 / 95.2 / 63.6 |

Reading:

1. The control reproduces the deployed float model (97.65 % / 95.46 %) within
   0.2 pp accuracy and 0.5 pp macro-F1. Its normal recall (95.8 %) differs from the
   deployed model's (87.2 %, CI 79.9–93.6): with 10 normal test files the normal
   recall varies strongly between training runs.
2. **Unseen speeds inside the trained range** cost nothing (97.9 %).
3. **Unseen intermediate severities** (interpolation) keep 95.7 % accuracy; the
   macro-F1 drop comes mostly from the normal class (79 %) and misalignment (91 %).
4. **Unseen highest severities** (extrapolation) fall to 74 %. In `sev_max` the
   normal test files are the fastest recordings, a speed range in which training
   has fault files only; normal recall 1.4 % is that speed extrapolation, not a
   severity effect. `sev_max_nmid` holds normal out as in sev_mid and isolates the
   severity effect: accuracy 74.1 %, macro-F1 72.4 %, normal recall 79.1 % — about
   22 pp below the middle-severity hold-out, lost mainly on bearing faults (63.6 %)
   and imbalance (77.9 %). Its best validation macro-F1 came at epoch 1: longer
   training fits the training severities more tightly and extrapolates worse.

Claim for the paper: the classifier generalizes to new recordings, unseen speed
bands and unseen intermediate severities of known fault types; it does not
extrapolate to severities beyond the training range.

## 5. Noise-augmented training

Training windows get white noise at SNR ~ U(0, 30) dB in half of the batches
(applied in the z-scored, 4:1-aggregated domain). Both float models are
evaluated on the exact raw-domain noisy test windows of the robustness sweep
(`generalization_20261110/noise_eval.json`; figure
`docs/figures/paper/noise_robustness`).

| SNR | control acc / normal recall % | noise-augmented acc / normal recall % | deployed INT10 (= board) acc / normal recall % |
|---|---|---|---|
| clean | 97.47 / 95.8 | 97.29 / 96.8 | 97.77 / 87.2 |
| 20 dB | 97.44 / 95.3 | 97.28 / 96.7 | 97.73 / 85.9 |
| 10 dB | 96.99 / 83.3 | 97.07 / 96.3 | 96.77 / 69.2 |
| 5 dB | 93.41 / 32.3 | 95.99 / 93.8 | 93.16 / 25.7 |
| 0 dB | 79.32 / 0.0 | 89.48 / 89.3 | 82.78 / 0.3 |
| −5 dB | 52.34 / 0.0 | 61.05 / 94.1 | 62.63 / 0.0 |

Noise augmentation removes the collapse of the normal class down to 0 dB at a
cost of 0.2 pp clean accuracy.

**INT10 version (2026-11-11; second bitstream on the board, see below).** The augmented
checkpoint was exported with the deployed flow (`python -m
src.quantize.export_hls_seq256_int10 --ckpt …/random_noise/best.pt --hls-dir
artifacts/quantized/f_seq256_int10_noiseaug --config-dir …`; same 20 training
calibration files). Its input scale equals the deployed one, so the on-FPGA
preprocessing constants are unchanged and the model would run on the raw-input
core with new weights only. The g++ C-model of the raw-input top with these
weights equals the Python INT10 reference on 300 clean and 300 0-dB windows
(logits and class register; `tools/check_variant_cmodel.py`). Full test, same
perturbed windows (`artifacts/results/robustness_noiseaug_int10_20261111/`):

| Condition | deployed INT10 acc / normal recall % | noise-augmented INT10 acc / normal recall % |
|---|---|---|
| clean | 97.77 / 87.2 | 97.40 / 96.6 |
| 20 dB | 97.73 / 85.9 | 97.40 / 96.5 |
| 10 dB | 96.77 / 69.2 | 97.19 / 95.7 |
| 5 dB | 93.16 / 25.7 | 96.19 / 93.8 |
| 0 dB | 82.78 / 0.25 | 90.13 / 88.8 |
| −5 dB | 62.63 / 0.0 | 62.75 / 93.1 |
| dead bearing-1 sensors (ch 1 / 2 / 3) | 80.4 / 76.5 / 77.9 | 82.0 / 79.0 / 73.4 |
| dead bearing-2 sensors (ch 4 / 5 / 6) | 89.6 / 90.2 / 90.4 | 90.9 / 89.8 / 92.5 |
| dead microphone / tach + mic | 84.2 / 78.1 | 86.8 / 80.3 |

The augmented INT10 model gives up 0.37 pp clean accuracy (macro-F1 95.67 →
94.92 %) for a much more stable normal class (96.6 % clean, 88.8 % at 0 dB) and
+7.4 pp accuracy at 0 dB. It does not fix dead sensors (not trained for).

**On the board (2026-11-11, MEASURED-PHYSICAL).** Second bitstream = the same
raw-input + vote core with these weights (`build_raw_core.ps1 -Variant votena`;
WNS +0.035 ns, 36,788 LUT; deployment `seq256_vote_noiseaug_z7020_20261111`;
the approved model stays the deployed one). Both bitstreams also ran a 0 dB copy
of the raw test input (`tools/make_noisy_testset.py`, the sweep's exact noise):

| Board run (95,404 windows) | per-window accuracy | system accuracy, FPGA vote k = 5 | bit-exact vs C-model |
|---|---|---|---|
| deployed model, clean | 97.7705 % | 99.084 % | 95,404 / 95,404 |
| noise-augmented, clean | 97.4016 % | 98.767 % | 95,404 / 95,404 |
| deployed model, 0 dB | 82.7827 % | 83.941 % | 95,404 / 95,404 |
| noise-augmented, 0 dB | **90.1304 %** | **93.835 %** | 95,404 / 95,404 |

Evidence: `artifacts/results/raw_core_board_fulltest_20261111_{130927,155757,170226,163200}/`.
At 0 dB the vote cannot rescue the deployed model (its normal class is lost, so
some recordings are never classified correctly), while it adds +3.7 pp to the
noise-augmented model.
At −5 dB its high normal recall comes with low accuracy (noisy fault windows are
called normal), so −5 dB is outside the useful range of both models.

## 6. Temporal vote in FPGA logic — status

- Implemented in the raw-input core (`temporal_vote.h`; AXI-Lite `vote_k` in,
  `vote_class` / `vote_count` out; continuous stream, ties to the newest window).
  Driver, board runner and full-test checker read the registers
  (`pynq/fault_classifier.py`, `tools/board/fpga_live_runner.py`,
  `tools/raw_core_board_test.py fulltest --vote-k 5`).
- First build (32-bit `int` vote logic): +7.3 k LUT after synthesis (85 % of the
  device), routing congested (WNS −1.44 ns after the first router iteration);
  stopped (`artifacts/results/seq256_dma_vote_z7020_20261110/first_build_int32_aborted/`).
- Second build (2-bit classes, 4-bit counts, but an early-exit `found` flag in
  the pick loop): HLS emitted control flow (top FSM 10 → 26 states) and the IP
  synthesized to 51 k LUT, all of the extra 16 k in the top module; stopped
  (`…/second_build_flagpick_aborted/`).
- Third build (pick as a select chain, oldest → newest): C++ native and ap_int
  builds = Python on 5,000 random steps; HLS C-simulation 41/41 logits + class
  and 41/41 votes; **post-route 36,911 LUT / 36,349 FF / 210 DSP, +241 LUT and
  +186 FF over the raw core, WNS +0.135 ns** (`artifacts/results/seq256_dma_vote_z7020_20261110/`,
  deployment `artifacts/deploy/seq256_vote_z7020_20261110/`). Lesson for the
  paper: HLS coding style decided a 70× difference in the cost of the same vote.
- **Board validation (2026-11-11, MEASURED-PHYSICAL):** golden 41/41; full test
  of 95,404 raw windows as one stream: logits and class register = C-model on
  every window (accuracy 97.7705 %), `vote_class`/`vote_count` at k = 5 = Python
  on all 95,404 windows, **system accuracy 99.084 %**, vote correct after median 2
  (41 ms) / max 4 windows after each of the 232 true-class changes; hard real time
  0 misses in 30,000 (max 16.66 ms); die max 55.6 °C
  (`artifacts/results/raw_core_board_{golden_20261111_130854,fulltest_20261111_130927,rt_20261111_133749}/`).
  The per-recording analysis (99.56 % at k = 5) excludes the first k−1 windows
  of each file; the continuous-stream number includes the transitions.

## 6b. Ten-class study (2026-11-11, Kaggle T4, float)

Same architecture, recipe and file split as the deployed model (split asserted
by SHA-256 of the file lists; test = the same 391 files / 95,404 windows), with
10 file-level classes: normal, imbalance, horizontal / vertical misalignment,
and ball / cage / outer-race faults on the underhang (U) and overhang (O)
bearings (`kaggle/seq256_10class/`, evidence
`artifacts/results/tenclass_kaggle_20261111/seq256_10class/summary.json`).

| Scoring of the same predictions | accuracy % | macro-F1 % |
|---|---|---|
| 10 classes (macro AUROC 0.9993) | 96.92 | 96.60 |
| collapsed to the 6 MAFAULDA groups | 97.65 | 96.41 |
| collapsed to the project's 4 classes | **98.31** | **96.55** |
| deployed 4-class float model, for reference | 97.65 | 95.46 |

Recall per class: normal 95.9, imbalance 96.3, horizontal 95.7, vertical 96.5,
U-ball 99.7, U-cage 96.9, U-outer 97.7, O-ball 99.5, O-cage 92.9, O-outer 99.0 %.
Largest confusions: overhang cage → overhang outer race (442 windows), imbalance →
underhang cage (357), vertical → horizontal misalignment (308).

Reading: the finer labels cost nothing; scored as 4 classes they are even better
than the 4-class training (single seed — the normal-class recall varies between
runs, see §4). The hardware cost is the classifier's last layer (16×4 → 16×10)
plus small interface changes (logit count, class width in the vote). A 10-class
INT10 export and bitstream would be a further paper study.

## 7. Still needed (board available until ~2026-12-09)

1. Measured power/energy (board adapter 5 V / 3 A): inline DC power meter (idle, overlay loaded,
   sustained inference; energy per inference = ∫P dt / windows).
2. Endurance: run 1 done (300,000 windows, 0 misses, all outputs = C-model);
   run 2 of 1,400,000 windows (~8 h) started 2026-11-11 17:33, results pending
   (`python -m tools.raw_core_board_test rt-fetch --run raw_core_rt_20261111_173322`).
3. Optional model work for the paper: per-channel amplitude (log σ) as a side
   input for the normal class.
