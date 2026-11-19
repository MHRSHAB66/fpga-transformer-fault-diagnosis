# Trustworthy diagnosis on the available Z-Turn: research directions beyond the existing checklist

Research cut-off: 2026-11-11, Asia/Tehran. Read-only investigation: no training, board job, model export, split change, or HLS change was executed. Effort ranges below are engineering estimates, not measured runtimes. Proposed success thresholds are examples to freeze before a pilot, not claimed results.

## 1. What is already tried, and what does recent primary literature actually leave open?

### Takeaway

Temperature calibration, dead-channel tests, highest-severity hold-outs, noise augmentation, and the bare idea of adding per-channel sigma are already present locally. The stronger opportunity is a **measured separation of machinery uncertainty, sensor-input invalidity, and finite-precision instability**, implemented as a cheap FPGA quality/abstention layer and evaluated against realistic fault families that were not used for training.

### Cited Findings

Local evidence, inspected this turn:

- `docs/reports/paper_readiness_review_20261110.md:40–44,202` already proposes preserving amplitude through sigma side information; it is not a new suggestion from this search. The same report records zeroed-channel robustness, noise augmentation, and severity generalization.
- `tools/fit_confidence_temperature.py:1–22` already fits a single positive temperature on validation integer logits, then evaluates test logits. This preserves the FPGA argmax and is already an existing confidence-calibration baseline. `tools/evaluate_confidence_calibration.py` already reports ECE, NLL and related metrics.
- `tools/robustness_cmodel.py:8–15` implements AWGN and whole-channel zeros. Its inspected corruption path does not implement asynchronous channel lag, intermittent holds, gain drift, or a burst occurring at a particular window position. Those are meaningfully different operators, rather than another Gaussian-noise level.
- `tools/generalization_study.py:1–26` already has `sev_mid`, `sev_max`, `sev_max_nmid`, and speed-band hold-out logic. The reported 74.09% highest-severity control is float, one seed, and already observed; it is not an untouched future final test.
- `src/config.py:81–107` sets 1024 samples, stride 1024, 50 kHz, per-window/per-channel z-score. Hence the raw window is 20.48 ms. This numerical fact is verified locally, not taken from a paper.

Primary works screened (10 distinct works; publication timing verified where available):

| Work | How far actually inspected | Useful supported result and limitation |
|---|---|---|
| Liang, Peng, Sun, **Selective Classification Under Distribution Shifts**, TMLR 2024 | Full HTML methods §§3.1–3.2 and experiments summary | Top-two raw-logit margin avoids changes in confidence ranking caused solely by global logit scaling; selection quality must be measured by risk–coverage. Their experiments are not machinery/INT10 evidence. [Paper](https://arxiv.org/html/2405.05160v2) |
| Windmann et al., **Benchmarking Sensor-Fault Robustness in Forecasting**, 2026, v2 dated 3 November | v1 full HTML taxonomy, stuck-sensor definition and protocol; v2 abstract and official repository; v2 HTML timed out on follow-up | Input faults span value, timing and availability. Train-fault families are disjoint from evaluated fault families; relative degradation and absolute fault-time error are separated. Forecasting findings do not transfer as measured classification improvements. [Paper/version history](https://arxiv.org/abs/2605.10822), [official repository](https://github.com/awindmann/SensorFault-Bench) |
| Ma et al., **Robust fault diagnosis method for failure sensors**, 2024 | Publisher English abstract, architecture and result tables; linked full HTML unavailable | CNN plus Transformer corrupted-feature attention masks already exist for diagnosis, including contiguous missing-data analysis. Therefore “mask broken sensors with attention” alone is not an original research contribution. [Publisher](https://www.zjujournals.com/eng/CN/Y2024/V58/I7/1488) |
| Wu and Lin, **Segmentalized amplitude normalization… under varying speeds**, online September 2024 / volume 2025 | Publisher full article abstract and introduction | Fault-related vibration energy and speed-dependent amplitude are coupled; order tracking plus carefully designed amplitude normalization is established prior art. It does not establish that adding raw RMS helps our classifier. [Publisher](https://journals.sagepub.com/doi/10.1177/14613484241277312) |
| Tu et al., **Integrating Frequency Guidance into Multi-source Domain Generalization for Bearing Fault Diagnosis**, 2025 | Full HTML motivation/method outline | FARNet separates Fourier amplitude/phase processing and uses training augmentation to address unseen working conditions. Generic Fourier augmentation is established; their architecture is not our tiny Transformer. [Paper](https://arxiv.org/html/2502.00545v1) |
| **Conformal prediction for out-of-distribution time-series classification**, Applied Intelligence 2025 | Full publisher §§1.2,2.1–2.2 | Consecutive gaps are a distinct distortion; standard exchangeability-based coverage is disrupted under shift. The work investigates label smoothing and k-NN/neural hybrids. Hardware suitability and guarantees for correlated MAFAULDA windows are not established. [Publisher](https://link.springer.com/article/10.1007/s10489-025-06708-7) |
| Kasa et al., **Adapting Prediction Sets to Distribution Shifts Without Labels**, 2024/2025 version | Full HTML introduction and method structure | ECP/EaCP use model uncertainty to adjust prediction sets on shifted unlabeled data. This motivates empirical comparison, not a guarantee under arbitrary unlabeled shift. [Paper](https://arxiv.org/html/2406.01416v2) |
| Franco et al., **Guaranteeing Robustness Against Real-World Perturbations In Time Series Classification Using Conformalized Randomized Smoothing**, UAI 2024 | Official proceedings abstract only, not full PDF | Transformation-specific randomized smoothing is established, but sampling cost must be checked. No runtime claim for this board follows. [Proceedings](https://proceedings.mlr.press/v244/franco24a.html) |
| Chau et al., **Generalizability Under Sensor Failure: Tokenization + Transformers Enable More Robust Latent Spaces**, 2024 | Full short HTML paper, introduction/method | Sensor-agnostic tokenization and Transformer sensor-failure generalization are already researched in EEG. Changing our tokenizer to a learned codebook would be a sizeable and weakly justified detour. [Paper](https://arxiv.org/html/2402.18546v2) |
| Mousanejad Jeddi and Nunez-Yanez, **Confidence Calibration in Quantized Graph Neural Networks Using Temperature Scaling**, 2025 institutional poster | Complete one-page PDF | Quantized-confidence calibration with validation-fitted temperature already exists outside LLMs. This is a poster and GCN evidence, not a strong independent basis for asserting INT10 Transformer confidence failure. [Primary institutional PDF](https://www.itn.liu.se/~marfa45/wasp/winterconf2025/posters/hadi_mousanejad_jeddi.pdf) |

Six works were read beyond their abstracts: Liang; Windmann v1; Wu/Lin introduction; Tu methods outline; Applied Intelligence §§1–2; Kasa introduction/method structure. Ma was inspected at publisher-abstract/table level, Franco at proceedings-abstract level; these should not be described as fully read. Failed accesses were retained as limitations. This is targeted screening, not a systematic exhaustive novelty search.

Search queries actually used included: `machinery fault diagnosis sensor missing channels uncertainty 2024 2025`; `bearing fault diagnosis normalization amplitude information domain generalization 2024`; `quantization calibration confidence out of distribution 2024 2025 neural networks`; `fault diagnosis sensor failure 2024 deep learning multi sensor`; `fault diagnosis missing 2025 sensor transformer`; `conformal prediction time series distribution shift 2024 2025 classification abstention`; `selective classification logit margin softmax response 2024 2025`; `Selective Classification Under Distribution Shifts Liang 2024 2025 arxiv`; `sensor time series failure corruption benchmark 2024 2025 neural classification`; `bearing order tracking FPGA diagnosis 2024 2025`; `Quantization confidence calibration neural networks FPGA classification -LLM -language`. General searches returned much LLM material; those results were not used to assert machinery behavior.

### Inferences

- “Zeroed bearing-1 channel hurts more” establishes diagnostic feature sensitivity to a synthetic removal. It does not prove that bearing 1 caused the mechanical fault, identify the failed component causally, or establish sensor health.
- A confidence score is not an input-validity detector. A clipped but highly class-like signal can be confidently wrong, while a healthy but ambiguous waveform can be low-confidence. Keeping these two flags distinct is a useful systems research question.
- Do not advertise a first-ever trustworthy FPGA classifier based on this screening. A defensible contribution would be the measured combination of a specific integer signal path, quality monitor, cheap rejection logic, realistic fault-transfer protocol, resource cost and deadline behavior.

### Gaps

- No inspected work establishes that our proposed quality flags, margin threshold, or sigma feature will improve this dataset/model. Every such benefit remains a falsifiable hypothesis.
- No native physical sensor front-end data, sensor gain tolerances or live machine condition transitions were established in the inspected local evidence. Raw24 input is modeled ADC code; this limits all field-deployment language.
- Some Springer exposition asserts a stronger APS class-conditional coverage claim than should be repeated without checking the original APS theory. Use only its clear exchangeability/gap-distortion discussion; do not inherit that claim.

## 2. Which four hypotheses are worth pursuing, in what order?

### Takeaway

Start with a no-retraining software pilot for fail-aware decisions and a physical-front-end transfer envelope. These can preserve the deployed classifier. Pursue amplitude-aware or training changes only if cheap validation-only pilots show a meaningful benefit over those baselines.

### Cited Findings

- A raw top-two margin is an established selection baseline and can be evaluated using the existing four integer logits; the literature does not justify skipping it in favor of ensembles or a novel rejector. [Liang et al.](https://arxiv.org/html/2405.05160v2)
- Timing and availability faults are separate from value corruption in the recent controlled benchmark protocol. [SensorFault-Bench](https://arxiv.org/abs/2605.10822)
- Amplitude can encode both defect and operating condition, so recovering it is not automatically beneficial. [Wu and Lin](https://journals.sagepub.com/doi/10.1177/14613484241277312)

### Inferences

#### H1 — Best immediate research direction: an FPGA that knows when its result should not be trusted

**Question:** Can a small pre-normalization input-quality monitor plus an integer top-two-logit margin substantially reduce confidently incorrect outputs under unseen sensor faults while preserving useful coverage and reaction time?

**Mechanism:** Keep the four-class FPGA argmax unchanged. Add separate versioned outputs `input_quality` and `decision_status`; they are system status, not a fifth machinery class. Start with channel-specific counters for rail occupancy, long identical-value runs, near-zero variance and discontinuities, then compare: (A) no rejection; (B) existing temperature-scaled maximum softmax; (C) raw integer `largest_logit-second_largest_logit`; (D) quality flags alone; (E) flags plus margin. Add conformal sets only as a secondary comparator. A threshold comparator and small counters are plausible PL additions; actual resource/timing costs must be measured, not assumed free.

**Critical controls:** Tachometer is naturally pulse-like, so a zero/flat tach segment inside 20.48 ms is not automatically a dead sensor. Constant acceleration can also be a valid low-energy condition. Fit channel-specific rules on training data, set thresholds using files in the validation split, and report clean false-invalid rates. Compare noise-augmented checkpoint to deployed checkpoint without claiming one is trained for sensor failure. Predicted-normal acceptance needs explicit scrutiny: rejected windows must never be silently converted to healthy.

**Cheap disconfirming pilot (0.5–2 days estimated):** Existing clean train/val logits plus a small deterministic raw-val corruption pack. Use full files as independent units; choose 3–5 prespecified severities, multiple fault positions, and paired clean/corrupt windows. Compute risk–coverage curves and false-healthy rate. A reasonable preregistered gate: at ≥90% clean coverage, reduce wrong accepted outputs under corruption by ≥30% relative versus no rejection, with no material increase in undetected-fault rate at the chosen coverage. The exact thresholds depend on intended operational cost; they cannot be retrospectively tuned on test.

**Kill criterion:** If simple flags/margin cannot separate errors at useful coverage, do not invent a complicated confidence head immediately; publish the limitation or pivot to H2/H3. If margin alone dominates combined flags, deploy the simpler margin and describe quality flags as observability, not improved diagnosis.

**Board evidence (2–5 additional days estimated):** Synthesized versioned status output, bit-exact counters/margin vs C-model, directed boundary vectors, corruption replay bursts through raw-input path, resource delta and worst-case latency/deadline misses. Also measure consecutive rejection duration, time to resume acceptance and accepted wrong decisions during burst onset/recovery. One inference per window preserves the ~20.48 ms budget better than repeated smoothing.

**Paper strength:** Moderate-to-high if the system actually avoids confident errors on disjoint fault types at low overhead. A clean ECE plot alone is weak. No blanket safety guarantee or unknown-fault detection claim.

#### H2 — New physical-system question: identify the real front-end transfer envelope of integer normalization

**Question:** Which acquisition distortions are removed by normalization, which are amplified or hidden, and where does the integer FPGA decision become unreliable?

**Mechanism:** In exact arithmetic, per-channel z-score removes `a*x+b` for constant positive `a` (subject to its epsilon policy); negative gain flips sign. Our rounded integer mean, variance, reciprocal approximation, raw24 rails and quantized input limit this invariance. Test pre-normalization positive gain and DC offset jointly with clipping, reduced effective ADC bits, intermittent holds, and channel-specific lag. This is not another AWGN sweep. Separate two outputs: deviation from the approved numeric reference and true-label error; a perfectly bit-exact implementation can remain diagnostically wrong.

**Cheap pilot (1–2 days):** On a small fixed training/validation raw subset, map logit drift, class flips and quality flags versus gain/offset/rail occupancy. In an unclipped affine control, expect strong invariance, so label flips near integer variance floors are particularly informative. Compare identical corruptions before versus after normalization; post-z-score corruption cannot model ADC clipping faithfully. Use both frozen deployed and noise-augmented models.

**Timing corruption:** Shift one accelerometer relative to the rest, freeze a channel for part of a window, and jitter sample availability. Keep the ground-truth machinery class fixed because the corruption targets acquisition, not the underlying fault. Do not shift every channel identically and call it sensor desynchronization. A common time shift is a separate phase nuisance control.

**Harder physical pilot:** If appropriate equipment exists, loop an externally generated multichannel/ADC-like digital stream through a documented board ingress, with programmable gain/range/clipping/lag. This tests ingress and quantization, not real sensor generalization. An analog ADC + signal source could evaluate an actual front-end; GPIO replay without ADC cannot substantiate analog performance. Never connect unknown voltage levels or assume an ADC exists.

**Kill criterion:** If all realistic parameter ranges are invariant and only absurd corruption hurts, this is useful validation but likely a subsection, not the paper's central contribution. If a clear transition appears, H1 flags should detect it; quantify false-invalid/false-valid tradeoffs at the transition.

**Board-only value (2–5 days estimated without new ADC):** Freeze and replay a compact boundary suite while hardware is available; capture exact input words, logits, status flags, timing and build hashes. Most severity sweeps and statistical analysis can follow after return. With new analog equipment, effort and procurement are unbounded until the actual interfaces are inventoried.

#### H3 — Strong model hypothesis, but sigma itself was Claude's idea: preserve fault amplitude without learning sensor-gain shortcuts

**Question:** Does restoring selected amplitude information improve normal/low-severity diagnosis and severity extrapolation, or does it simply make the classifier sensitive to calibration and speed?

**Mechanism:** Two-path research variant: current normalized eight-channel Transformer plus a small fixed-point feature branch using per-channel `log sigma` or a coarse exponent, optionally robust relative vibration-amplitude ratios. Do not change canonical tensor conventions or replace the deployed model. Keep all eight channels and four grouped output classes. The existing variance computation may be reused, but routing, feature scaling, parameters and classifier integration are new architecture work requiring explicit research-variant documentation. “Almost no hardware” is not established.

**Smallest useful pilot (1–3 days):** First compute sigma-only and simple RMS-ratio probes on train/val. Then freeze the current float backbone, extract its pooled embeddings, and fit equal-regularization linear heads for embedding-only vs embedding+8 log-sigma. This cheaply estimates whether the lost amplitude contains incremental information. Partition the existing validation files internally into model-selection and calibration subsets; do not fragment windows from a file. Do not use test results to select features or regularization.

**Pilot metrics/gate:** Prespecify normal-vs-fault AUROC and normal/low-severity recall at matched false-fault rate, plus original four-class macro-F1; bootstrap by file. Example advance gate: ≥3 percentage-point validation normal recall improvement at matched false-fault rate, ≤0.5-point macro-F1 regression, and benefit surviving independently applied gain nuisance. These are proposed gates. A sigma-only classifier doing extremely well is a warning to inspect speed/condition confounding, not proof that the Transformer is unnecessary.

**Controls that make it more than the old sigma suggestion:** Stratify by speed; compare relative-amplitude features to absolute sigma; independent positive-gain perturbations affecting vibration channels; sensor offset/clipping; original frozen checkpoint; noise-augmented checkpoint; same architecture retraining without side info. Train augmentation and scored fault families must be disjoint. Evaluate known observed severity hold-outs as exploratory, then reserve a prespecified additional condition or data source for independent confirmation if available. No new split replaces canonical published results.

**Kill criterion:** If improvements disappear after speed matching or modest gain perturbation, do not deploy raw amplitude side info. If frozen-embedding pilot gives no meaningful incremental benefit, skip costly retraining/bitstream until there is a better mechanism.

**Board stage (roughly 1–2 weeks after a successful pilot):** Quantize the side feature branch, compare float to exact integer, regenerate new versioned goldens, synthesize variant and test raw24 path, amplitude-range/quality-gate boundaries, and resource/deadline effects. The board is needed for the final implementation evidence; early model search is software work.

#### H4 — Ambitious option: operating-condition invariance taught during training, with unchanged inference cost

**Question:** Can tach/frequency-informed training improve generalization of the same small time-domain Transformer without imposing order tracking or a second network at inference?

**Mechanism:** A training-only teacher or consistency loss can use longer training-file context, operating speed, and physically justified resampling/phase transforms, while the deployable student remains the existing 1024×8→256-token architecture. Keep it a separate research checkpoint. Compare to ordinary consistency/augmentation and the already available noise-augmented baseline. A full FARNet/learned tokenizer on board is not the preferred first step.

**Important feasibility constraint:** A 20.48 ms window at 50 rotations/s contains ~1.024 shaft revolutions. At lower speed it contains less than one. Inferring precise RPM from two tach edges inside every independent window is not generally reliable; verify pulses-per-revolution, edge shape, acquisition semantics and speed range first. File-name speed is permitted as training-analysis metadata, but it is an oracle if a deployed method claims to receive speed without measuring it. A stateful long-context tach estimator is another explicit variant with startup delay and missing-pulse handling.

**Cheap pilot (2–4 days):** First stratify existing validation errors by speed and check tach edge statistics on contiguous raw training/validation files. Test whether common channel-synchronous time scaling within a small physically justified range preserves the grouped label and improves nearest-neighbor/frozen-embedding consistency. If the signal transform destroys fault impulses or low-speed contexts contain too little periodic information, stop. A short paired training experiment on identical data/seed with and without a training-only consistency term is the next gate; repeat successful variants over ≥3 seeds before making accuracy claims.

**Kill criterion:** No validation benefit beyond simple time augmentation, instability across seeds, or reliance on target-test speed/statistics. Do not use test-time gradient adaptation as the initial route; labels may be absent and a drifting model complicates exactness and safety claims.

**Board stage (1–3 weeks estimated overall, highly training-dependent):** If unchanged architecture wins, export weights through existing approved flow, verify exactness, then measure latency/resource and worst-case raw-input/quality behavior. The central novelty must be a successful physically grounded student plus measured integer deployment; domain generalization by itself is crowded prior art.

### Gaps

- Equipment availability and current resource slack are unknown here; inspect post-route build and available ingress before promising physical H2 or H3 cost.
- MAFAULDA supplies conditions/recordings, not labelled real transition-onset times. Concatenated file transitions can measure pipeline response and rejection recovery, but cannot establish actual machinery-fault onset latency.
- An unknown fault class is unavailable in the current four-class taxonomy. Unseen severity or corrupted known-class inputs are covariate shifts, not new-class OOD. Do not present them as unknown-fault discovery.

## 3. What must be captured before the board returns, and how should conclusions be bounded?

### Takeaway

Prioritize hardware evidence that cannot be recreated later: versioned bitstreams, raw-input boundary cases, exact status/logit traces, timing and recovery. Long numerical sweeps, plotting and most training can continue after the board is gone. Preserve an evidence boundary between simulated corruption, digitally replayed hardware input, and actual sensor acquisition.

### Cited Findings

- Standard conformal guarantees depend on exchangeability; unlabeled shift adaptation is a separate research problem rather than automatic protection. [Kasa et al.](https://arxiv.org/html/2406.01416v2)
- Prediction sets that include every class can achieve coverage while being operationally useless; report set size alongside coverage. [Applied Intelligence paper](https://link.springer.com/article/10.1007/s10489-025-06708-7)
- Recent sensor-fault benchmarking explicitly limits robust conclusions to the evaluated scenarios and calls for deployment-recorded fault validation. [SensorFault-Bench v2](https://arxiv.org/abs/2605.10822)

### Inferences

Protocol to freeze before any new evaluation:

1. Hash deployed model/scales/bitstream, canonical file lists, raw24 converter, fault operators and scenario seed. Preserve the ongoing endurance job; no concurrent board tests.
2. Build a small corruption suite on train/val first. Selection uses validation files only. Internal calibration/threshold-fitting folds are file-disjoint; do not treat 95k correlated windows as 95k independent calibration observations. Window-level conventional CP needs a justified exchangeability unit; file/recording-level empirical coverage is safer to report without a theorem.
3. Separate two hypotheses: detecting a sensor defect and reducing machinery classification errors. Use sensor-quality labels only for synthetic injected defects, true four-class labels for classification. Report input-defect detection precision/recall, accepted risk vs coverage, per-class normal/fault false decisions, rejection duration and recovery delay.
4. Thresholds selected at float precision may migrate after quantization. Fit/validate on deployed integer logits. Compare float vs integer: argmax agreement, top-two margin/ranking changes, rejection agreement and confidence near threshold. A cheap margin baseline may beat softmax/entropy; do not assume any fancy detector wins.
5. Use paired file bootstrap or per-file distributions for uncertainty. Distinguish relative error reduction from percentage-point accuracy change. Report worst scenario as well as average; hide neither rejected windows nor already-seen test exploration.
6. For every hardware variant capture a small exact raw24 input pack covering clean, gain/offset near the invariance edge, rail saturation, low variance, burst-hold onset/end, timing lag, confidence ties/threshold neighbors, and reset/startup/recovery. Store integer logits, status fields, frame IDs, measured deadline boundary, drop counts and software reference outputs.
7. Compare inference-only and raw-window-to-status boundaries separately. Do not add averaging/ensembles without including their extra time and information-accumulation delay. Randomized smoothing with many forward passes is currently a poor default for a ~16 ms core inside a 20.48 ms window interval; it requires a new throughput strategy or a relaxed decision rate.

Practical order: **H1 pilot → H2 boundary map → only then H1 small hardware monitor; H3 pilot in parallel software time → H3 board variant only if it passes; H4 as an ambitious reserve.** Basic AWGN, zero-channel, temperature-scaling and longer endurance repeats remain controls/evidence completion, not the new main scientific story.

Potential paper framing if results support it: **A numerically exact, input-aware selective Transformer for machinery monitoring on a small Zynq FPGA, with realistic sensor-fault transfer and a measured coverage–error–latency tradeoff.** If H3 also wins, a stronger mechanism is **amplitude-aware diagnosis that retains condition information without sacrificing acquisition robustness**. Both are hypotheses; a paper venue or acceptance probability cannot be promised from the present evidence.

### Gaps

- No results from these proposed pilots exist yet. This note recommends experiments and decision gates, not claims that any would succeed.
- No conformal guarantee is asserted under arbitrary sensor faults, severity shift or correlated streaming windows. Adaptive online conformal methods may require true labels after each prediction; availability of those labels is not established in this monitoring application.
- “All work while the board is available” should mean preserving scarce hardware evidence, not running every possible corruption until test-informed selection contaminates the result.
