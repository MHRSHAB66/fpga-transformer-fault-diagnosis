# Hardware co-design directions grounded in the actual INT10 core

## What is new relative to this repository, and what literature already establishes?

### Takeaway

The strongest coherent hardware research direction is **operator-specific numerical precision plus exact range-specialized arithmetic**, tested on the *actual small FPGA after place-and-route*. Generic INT8, pipelining, head parallelism, LUT nonlinearities, and integer-only inference are already established in the literature; several are also already implemented here. There is a concrete, reproducible numerical mechanism worth investigating: the current signed probability coding can map an entire diffuse attention row to zero at INT8 and 256 tokens. This is a new finding for this review, **not a claim of a new quantization principle**.

Review date: 2026-11-11. No board commands, hardware reloads, toolchain builds, training, production code edits, or memory updates were performed. The long-running board job was left untouched. Only these notes and tiny research evidence files were written.

Reproducible pilot: [script](hardware_pilot.py), [raw JSON evidence with source hashes](hardware_pilot.json). From repository root run `python -c "import runpy; from pathlib import Path; runpy.run_path(str(next(Path('research_notes').glob('*/hardware_pilot.py'))), run_name='__main__')"`. The range proof assumes approved clamped internal codes [-511,511]; representable `ap_int<10>` code -512 is outside that contract. If any alternative ingress permits -512, regenerate with a conservative512 magnitude or explicitly validate/reject it. Never take abs(MIN) in the same signed type: widen first.

### Cited Findings

**Primary literature screening record.** Thirteen selected works were screened at abstract or body level; seven had useful methods passages actually read, one additionally had its authors' implementation README read. “Read” below means the indicated relevant passages, not an assertion that every page was read. Publication dates are at or before the review date. Search queries included FPGA integer Transformer nonlinear mixed precision; HG-PIPE; small FPGA integer time-series Transformer; QFlash; RepQ/AdaLog; DSP48E1; FQ-ViT/ITA/Softermax. Search-service snippets were not treated as full-paper reads.

| Work | Actual inspection | Relevant evidence and limit |
|---|---|---|
| Ling et al., *Integer-only Quantized Transformers for Embedded FPGA-based Time-series Forecasting in AIoT*, 2024 | Body: architecture §§II–IV, divider discussion and experimental structure | Uses 4/6-bit integer/QAT on Spartan-7; replaces LN with BN and activation with ReLU; lower width does not consistently lower latency/energy. Thus its headline efficiency does not directly establish our unchanged LN/GELU classifier's optimum. [Primary paper](https://arxiv.org/html/2407.11041v2) |
| HG-PIPE, 2024 | Body §§4.3–4.4 | Balances pipeline stages and BRAM layout; GELU–requant fusion, table requantization, segmented reciprocal and careful exponential anchoring. These are concrete baselines, not inventions to claim. Its 64-entry table result belongs to its own calibrated network. [Primary paper](https://arxiv.org/html/2407.17879v1) |
| Quasar-ViT, ICS 2024 | Body §§4.2–4.3, pp.6–8 | Uses 4-bit atomic products, mixed precision and signed/unsigned DSP packing. Crucially its implementation describes **DSP48E2 with 27-bit port**, not the Zynq-7020 DSP48E1. Packing factors cannot simply be copied. [Authors' paper](https://zhenman.github.io/files/C36-ICS2024-Quasar-ViT.pdf) |
| TATAA, TRETS 2025 / arXiv version | Body precision design, §3, runtime §6.4 | Keeps linear operations INT8 and nonlinear operations bfloat16; nonlinear operations contribute about 25% runtime in its examined workloads. It demonstrates differing precision needs, not that we should add floating point here. [Primary manuscript](https://arxiv.org/html/2411.03697v1), [publisher record](https://hub.hku.hk/handle/10722/366101) |
| Kim et al., *Hardware Accelerator for Approximation-Based Softmax and Layer Normalization*, 2025 | Body §§4–5 | Shares a PE datapath, piecewise-linear exponential and Newton–Raphson square root; tunable approximation. Its 32-lane/32-bit VU37P architecture is not a ready-made fit for this board. [Publisher full text](https://www.mdpi.com/2079-9292/14/12/2337) |
| Xu et al., *An FPGA Accelerator for Vision Transformer with Quantization and LUT-Based Operations*, IEICE Jan 2026 | Body §3.2 | Different W2A6/W4A8/W8A8 array shapes, shared linear/attention processing elements; hardware-aware quantization is established. Its model, FPGA and task differ from ours. [Publisher full text](https://globals.ieice.org/en_transactions/information/10.1587/transinf.2025PAP0003/_f) |
| AdaLog, ECCV 2024 | Body §§3.1–3.3, pp.5–8 | Adaptive logarithmic probability coding; integer implementation via two tables and shifts; explicitly addresses unsuitable uniform probability bins. A useful comparator if simple unsigned/mixed precision fails, not a free accuracy guarantee. [Conference paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03969.pdf) |
| QFlash, IEEE TC early access, 2026-10-08 | Publisher abstract + authors' README, **not full paper** | Integer online softmax/attention fusion already exists. Authors describe row-granularity fused attention and separate attention, linear/GELU and AddNorm engines. The DOI page could not be fetched directly; do not assert detailed paper results from a full-text read. [Publisher record](https://xplorestaging.ieee.org/document/11721538/), [authors' released code](https://github.com/wzh0715/QFlash) |
| FAMOUS, 2024, revised 2025 | Abstract only | Dense MHA tiling/parallelism on U55C/U200, not a complete small-device classifier. [Primary abstract](https://arxiv.org/abs/2409.14023) |
| RepQ-ViT, ICCV 2023 | Abstract/search-visible conference text; full-PDF fetch blocked 403 | Scale reparameterization for LN and logarithmic post-softmax probability coding are prior art. Do not claim the method was fully read. [Conference PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_RepQ-ViT_Scale_Reparameterization_for_Post-Training_Quantization_of_Vision_Transformers_ICCV_2023_paper.pdf) |
| FQ-ViT, IJCAI 2022, foundational exception to 2023–2026 emphasis | Abstract and search-visible methods passage | Log-Int-Softmax already combines integer exponent approximation with logarithmic attention coding and bit-shift multiplication. Therefore “low-bit softmax” or “logarithmic attention” alone is not new. [Conference paper](https://www.ijcai.org/proceedings/2022/0164.pdf) |
| ITA, ISLPED 2023 | Primary abstract | Integer-only streaming softmax and 8-bit attention accelerator; published efficiency is for 22nm ASIC, not our FPGA. [Primary abstract](https://arxiv.org/abs/2307.03493) |
| Softermax, DAC 2021, foundational exception | Primary abstract | Base replacement, low precision and online normalization were already co-designed. No first-online-softmax claim is defensible. [Primary abstract](https://arxiv.org/abs/2103.09301) |

Official device constraint: DSP48E1 is a signed **25×18** multiplier with a 48-bit datapath; SIMD add modes do not create independent SIMD multipliers. Account for sign extension and guard bits when transferring any packing scheme. [AMD UG479, pp.17–18,29,40](https://docs.amd.com/api/khub/documents/gu4oRPFEh_Pm2uaAlfY6Kg/content)

**Repository evidence (read directly, not inferred from literature).**

- Current manifest: T=256, original window1024×8, pool factor4, D32, H4, FF64, one layer, activation/weight/attention10, exp_frac15. [Manifest](../../hls/f1_int10_seq256/model_manifest.json)
- The deployed-source family already has K banking, four head DATAFLOW, two key lanes, FFN unroll4, fused K/V projection, classifier fabric multiplication, and digit-based DSP-free raw normalization. [Current raw core](../../hls/f1_int10_seq256/vitis/transformer_top_seq256_raw1024_dsp_margin_argmax.cpp), [preprocessing](../../hls/f1_int10_seq256/vitis/onchip_preprocess.h)
- Prior `hls_throughput_study.md` is a **T1024 historical HLS-only study**. It cannot describe current T256 latency or be re-presented as a new experiment. T1 banking and T2 shared context requantization were already attempted. [Historical report](../../docs/reports/hls_throughput_study.md)
- Current raw deployment reports 210/220 DSP, LUT36,670 and BRAM76.5, WNS+0.167ns at100MHz. Wide multiplication mapping already caused route failures; fabric classifier requantization already exists. [Physical raw-input report](../../docs/reports/onchip_preprocessing_20261109.md)
- D014 uniform16/14/12/10/8 sweep chose INT10; INT8 imbalance recall0.66. That is an earlier-model result; it does not prove the cause or settle the current T256 mixed-precision design. [Decision log](../../docs/DECISION_LOG.md)
- The multibit evaluator explicitly sets activation, weight **and probability** widths together. `_softmax` uses `qint_max(ATTN_BITS)`, rounds each probability independently, has no mass restoration/zero-row fallback, then context uses the reciprocal qmax scale. [Integer model](../../src/quantize/integer_model.py), [coupled sweep](../../src/quantize/evaluate_integer_bits.py)

### Inferences

- Merely adding another uniform bitwidth curve, a generic softmax LUT, or more HLS pragmas would be weak and partly duplicate work.
- A credible applied co-design contribution could explain **why** a cheap numerical representation fails, repair the implicated operator while preserving the full Transformer, and verify the resulting accuracy/resource/latency Pareto on this constrained device.
- Exact arithmetic specialization is a lower-risk first implementation, but may yield only incremental engineering improvement. It becomes more publishable when combined with a principled precision allocation experiment and physical evidence.

### Gaps

- No literature result guarantees acceptable W8/A8 accuracy on MAFAULDA, or any DSP saving after our toolchain maps constants.
- QFlash full paper was not accessible through the attempted DOI fetch. We do not claim a comprehensive novelty clearance or absence of closer work.
- A source comment still says target is provisional; it is historical stale boilerplate, not current board status.

## Which three experiments most strongly distinguish useful research from extra testing?

### Takeaway

Do A first for physical opportunity, B for the most promising numerical research hypothesis, and C only if profiling shows it can beat A+B. A preserves exact approved outputs; B and C are **new numerical research variants requiring an explicit implementation choice**, separate exports and golden references. Neither silently replaces the original path.

### Cited Findings

The sources and repository links above motivate the proposals; all targets, effort ranges and go/no-go thresholds below are **proposed experimental criteria, not measured promises**.

### Inferences

#### A — Exact range-specialized arithmetic instead of generic wide requantization

**Hypothesis.** Known constant weights, INT10 clipping and fixed Q40 scale constants permit narrower, specialized arithmetic with identical integer outputs. Removing expensive generic operators may free DSP/routing capacity or give timing margin. HLS and Vivado may already perform some narrowing, so only the post-route result settles this.

**Mechanism specific to this code.** `requant(a,M)=clip511(round_even(a*M/2^40))`. For linear output o, legal pre-requant accumulator obeys `|a_o| <=511*sum_i|W_oi|+|b_o|`. This is a *legal-domain bound*, not observed min/max. We evaluated constants from the current checked-in `weights_int10.h` with a read-only Python script:

| Layer | Max legal bound incl bias | Sufficient signed bits | Q40 M unsigned bits | Full product sufficient signed bits | Safe early-saturation threshold T | Signed bits for non-saturated multiplier input |
|---|---:|---:|---:|---:|---:|---:|
| projection |719316|21|32|53|174084|19|
| Q |1748369|22|32|53|226467|19|
| K |1339961|22|32|53|241187|19|
| V |1760425|22|30|52|599111|21|
| attention output |2149582|23|31|53|442441|20|
| FF1 |2564229|23|31|53|513085|20|
| FF2 |3861416|23|31|54|380318|20|
| classifier1 |2203881|23|30|52|630982|21|
| classifier2 |1218424|22|31|52|396345|20|

Here `T=ceil(512*2^40/M)` for positive M. If `|a|>=T`, returning sign(a)*511 is provably equivalent because the un-clipped rounded magnitude is at least512. Otherwise retain exact multiplication and round-to-even. This threshold is deliberately conservative; it does not discard a possible boundary bin. Use comparison before casting to narrower type. Negative scales or future exports need regenerated proofs; do not reuse these bounds blindly.

**Minimal discriminating experiment.** Freeze current bitstream/source/weight hashes. Map synthesis hierarchy to actual expensive products, then change one high-cost linear/requant operator only. Compare (a) current generic, (b) exact per-layer widths/constant specialization, and optionally (c) exact saturation-aware version. Native/HLS CSim must match every logit on approved vectors and boundary-focused inputs before synthesis. Explore residual/LN only after separately proving centered input, rsqrt and affine bounds. A failed narrowing is not excused by unchanged class predictions.

**Fair baseline/measurements.** Same checkpoint, same weights, same Q40 values, same protocol, same clock target and same source of input. Post-route LUT/FF/BRAM/DSP, setup/hold, critical paths, actual hardware cycles and integer outputs. If any cycle penalty appears, compare at equal delivered throughput as well as equal clock. A valid shared unit may reduce DSP while consuming more cycles; show both rather than choosing the flattering metric.

**Proposed gate.** Proceed to an integrated new bitstream only if exact CSim holds and either total post-route DSP drops by at least10%, or timing/latency provides a material improvement (e.g. at least15%) without a compensating resource failure. Those are screening thresholds we can adjust before running. Stop after the targeted pilots if mapping is unchanged or route/timing gets worse; a negative result is useful, not grounds for an open-ended refactor.

**Board-only evidence before returning board.** Same frozen replay windows, exact logits+classes vs old board/reference; cycle counts, valid/result alignment and integrated timing at each supported clock. Capture a small checksum-trace set plus the already approved full fixed test only after selecting design on train/validation evidence. Archive final bit/hwh/report/source hashes. Repeated test evaluation is not a tuning oracle.

**Training/HLS/bitstream/effort.** No training. Yes new HLS export and post-route bitstream. Estimated **4–10 engineering days** including proof scripts, one-operator pilot, integrated mapping and board checks; estimate derives from existing native/CSim/deployment scaffolding, with constant-propagation surprises and HLS2020.2 typing as principal uncertainty. Tool runtime is additional and unmeasured.

**Failure modes/proof obligations.** Host `long` width differs from synthesized C++; introduce explicit `ap_int/ap_uint` widths and a separate arbitrary-precision oracle. Widen *before* multiply/add; prove accumulator and bias separately; preserve round-even and negative division/remainder semantics. Do not replace signed right shift with truncation toward zero. Do not shorten Q40 M merely because its printed value is small. Prove residual/LN triple products and every intermediate before narrowing. Removing output saturation or matching only top1 is insufficient. No claim that a generic64-bit product becomes one DSP: M itself needs30–32unsigned bits and DSP48E1 is25×18signed.

#### B — Attention-resolution-aware mixed precision, with a causal failure test

**Hypothesis.** The lower bound on usable *uniform probability resolution* depends on token count and attention diffuseness, not simply MAC precision. Current all-at-once INT8 may waste an attention sign bit and erase diffuse rows; separating probability precision from weight/linear precision could retain accuracy while reducing the dominant hardware cost.

**Repository-specific pilot already run.** Called the actual `IntegerTransformer._softmax` with equal integer scores and the exact exp(0)=32768 LUT value, no model edits or board involved. At T256: signed8 Qmax127 produces code0 in all256 entries and rowmass0; signed9 Qmax255 produces code1 and rowmass256; signed10 Qmax511 produces code2 and rowmass512. T1024 signed10 also produces all-zero rows. The general flat-row condition is `round_half_up(Qmax/T)=0` when `Qmax<T/2`. This is a deterministic boundary of this coding rule, not evidence that real data has flat rows or that it caused historical class collapse. There is no mask/causal prefix in the current classifier, so T=256 is the full denominator length. Masked attention variants would need the *actual unmasked length*, but adding masks is outside this experiment.

**Minimal causal ablation (train-calibration/validation only).** Freeze the current T256 checkpoint and calibration file IDs. Compare: fullINT10; uniformINT8; only weights8; only linear activation8; only probability signed8; probability signed9; **unsigned probability8 Qmax255** with correct context scale; and the best W8/A8 plus probability9/10 candidate. Keep LN internal range/exp Q15/Q40 baseline initially. Keep scores separate from q/k/v precision. This isolates whether the cheaper fix is assigning the nonnegative quantity its full unsigned range or whether additional probability precision is necessary. Do not compare a retrained candidate against an untrained quantization baseline without reporting that distinction.

**Measurements.** Per head, per validation file: all-zero row fraction, number of nonzero entries, normalized rowmass `sum(qattn)/Qmax`, pre/post-quantization entropy where defined, max probability, context error, saturation counts; per-class recall and macro-F1. Relate changed predictions to these numerical markers. Include flat, peaked, bimodal and near-threshold toy rows; do not infer causality from class accuracy alone. For finalists, CSim-reference exactness to the NEW numeric contract plus post-route full resource/cycle results on the physical board. W8 can have lower memory but identical DSP count unless compute mapping changes; signed/unsigned casts must be explicit.

**Prior-art baseline.** FQ-ViT LIS and AdaLog already preserve small attention values through nonuniform coding. If simple unsigned8 or mixed probability9 fixes the effect with lower cost, that is an attractive *task/device-specific engineering result*, not a novel quantization theory. If it does not, compare one logarithmic integer coding variant inspired by published methods before inventing a bespoke scheme. TATAA reinforces that linear and nonlinear precision should be separated, but its floating-point nonlinear path is not required here.

**Proposed gate.** Continue only if the ablation both identifies a meaningful numerical effect on real validation data and yields a candidate with, for example, macro-F1 loss<=0.3percentage points, no class recall loss>1point, and >=15% improvement in a resource/latency objective after route. Freeze criteria before seeing test results. If zero rows are negligible or do not explain errors, reject that causal explanation and inspect score/LN saturation; do not retrofit an underflow story. If resources do not improve, it may remain a software ablation rather than the paper's hardware contribution.

**Board-only evidence.** At least baseline, deliberately deficient uniform8 (if it safely fits), and selected mixed variant implemented/compared with identical protocol and input order, logging logits or documented output precision. Timing-valid outputs must match each variant's separate reference. Board is needed to establish actual resource/cycle improvement and correct implementation, not to manufacture a new accuracy number already implied by a verified reference. Select final model using validation; report held-out test once per frozen prespecified comparison, with file-level uncertainty.

**Training/HLS/bitstream/effort.** PTQ pilot requires no training. If QAT becomes necessary it is a separate prespecified branch on train only with repeated seeds and same split. New HLS types, export manifest, LUTs/scales and bitstreams required for finalists. Estimated **1–3 engineering weeks for PTQ-to-board**, **2–4 additional weeks if QAT/export changes are needed**; estimated because mixed scales affect attention, residuals and export, with multiple builds and numerical debugging, not because a timing experiment has been performed.

**Failure modes.** Historical sweep was on an earlier model; recalc the current one. Correcting one sign bit may improve rowmass without recovering class recall. Softmax probabilities can have rowmass≠1 even when nonzero; protect context accumulator bounds under worst-case *coded* sums. Unsigned8 must be cast correctly before signed V multiplication. Changing post-attention scaling or probability coding requires new golden outputs and explicit new-version naming. Keep all8 input channels, class map and wire protocol unchanged.

#### C — The hardware cost of intermediate rounding: deferred normalization/fused attention

**Hypothesis.** Current computation rounds each probability before multiplying V. A research variant that accumulates unnormalized weighted values and normalizes only at the end may remove many divisions and probability storage while reducing a distinct rounding error. It may instead require wider products/dividers and offer no net advantage; this is exactly what the experiment should discriminate.

**Mechanism and overlap check.** Current source uses row-local scores/num/qattn and already avoids the full T×T off-chip matrix. Therefore “we added FlashAttention to remove a huge external attention matrix” would be false. The fresh change would be specifically replacing `sum_j round(Qmax*n_j/d)*v_j` by a separately specified deferred-normalization calculation based on `sum_j n_j*v_j / d`. These expressions are **not equal**. Online max updates additionally change finite LUT/rounding semantics, so stage the work: fixed-row max + deferred division first, online tiled maximum only if justified. QFlash/ITA/Softermax already establish online/fused softmax prior art.

**Minimal experiment.** Native arbitrary-precision reference on training/validation compares current arithmetic, deferred division with the SAME exp table and rowmax, and one chosen online variant. Profile operator cycles first. Worst-case exp15 V10 sum256 needs explicitly derived wider signed intermediates; include rowmax transitions/tile boundaries/ties in reference tests. Use a single head micro-kernel HLS pilot to compare real divisions, RAM ports, multipliers and initiation interval before rewriting the whole core.

**Fair baseline/measurements.** Same trained weights, token length256 and exp LUT unless a declared later ablation changes them. Compare accuracy/macro-F1, logit/context deviation, per-head mass/error, post-route RAM/DSP/LUT, cycles, clock margin. Do not compare changed rounding to baseline golden outputs and call deviations bugs; each version has its own reference. Do not hide task-accuracy losses behind better exp-function MSE.

**Proposed gate/stop.** Stop if the isolated legal-width kernel has no credible integrated advantage, if fixed-row deferred rounding already violates prespecified accuracy limits, or if wider normalization consumes the saved DSPs. Continue only for a validated candidate offering >=20% latency or resource advantage at acceptable class metrics. These are proposed criteria. Do not build an entire copied large-board accelerator because a paper reports high GOPS.

**Board-only evidence.** At least one integrated timing-valid finalist, exact match to its *new* integer reference on boundary vectors and frozen replay; cycle and route reports; quality comparison on frozen test after selection. Board data is necessary for an actual speed/resource claim.

**Training/HLS/bitstream/effort.** Reference/PTQ first, retraining optional later. Substantial HLS and new bitstream necessary. Estimated **2–5 engineering weeks** because normalization order touches all heads, widths, division schedules and dataflow; online maximum can enlarge it further. Original architecture and external interfaces remain intact as the reference path; mathematical precision semantics change explicitly in the research version.

**Failure modes.** Per-element rounding cannot be algebraically moved through a sum. Dynamic row maxima with integer exponent tables are not exactly rescalable. Removing qattn storage can be trivial compared with K/V buffers. Existing design is DSP constrained, so widened n×V products may be worse. An online kernel can introduce variable stalls; average cycle reduction is insufficient if it breaks the prepared-window deadline.

### Gaps

- No trained-data attention histograms were computed in this review. The flat-row pilot is a real numerical result, not a MAFAULDA result.
- No new design has been synthesized/routed. Every saving target above is conditional.
- No training/hardware time limit or extra apparatus availability has been assumed.

## What should be collected while the board is available, and what should be rejected?

### Takeaway

The board should establish the **physical frontier of a few causally different designs**, not host dozens of variants with no hypothesis. A+B can form one defensible narrative: sufficient attention resolution plus exact hardware arithmetic enables a smaller/faster complete Transformer on XC7Z020. If the differences disappear after routing, report that honestly and keep the demonstrated full system as the engineering contribution.

### Cited Findings

The current raw system has narrow timing slack and near-full DSP usage; its preprocessing report documents that forcing one multiplier into fabric or consuming another DSP can break timing elsewhere. This is why operator-level HLS resource counts alone cannot establish the result. [Local physical implementation evidence](../../docs/reports/onchip_preprocessing_20261109.md)

### Inferences

1. Preserve current deployment source/bit/hwh/tool/config/calibration hashes and a short exact-logit fixture before any future reload. No reload was done in this review.
2. Run board-independent A proof and B numerical discrimination while current long job proceeds. These are future implementation tasks, not completed here.
3. Compile only finalists: baseline; A exact-specialized; B selected mixed; A+B if individually useful. Add C only if its kernel gate passes. Avoid a Cartesian product of every bitwidth/unroll/clock.
4. Compare at100MHz first, with actual setup/hold and integrated DMA/PS constraints. Then optionally build prespecified lower/higher **timing-closed** clocks to find a valid operating envelope. Post-route slack at100MHz is not measured Fmax. Do not call a runtime clock that exceeds timing closure reliable merely because a short replay succeeds. No voltage changes, overclocking or thermal stress are needed.
5. Report Pareto points using the actual post-route totals and actual window-cycle distributions. Keep throughput, first-result latency, acquisition20.48ms and host transfer boundaries separate. If power instrumentation becomes available, add physical energy/window with idle/active boundaries; otherwise do not substitute vectorless estimates as measured energy. The base plan requires only board+PC.
6. Capture build manifests and compact raw evidence while board exists: cycle counters/logits/class IDs for the fixed cases, board settings and observation boundaries, route reports, exact reference output hashes and failure logs. A complete source/bitstream archive permits later analysis, but cannot recreate unmeasured physical behavior after board return.

**Rejected as primary new hardware directions:** repeating K banking/head parallelism/FF unrolling already implemented; moving nonlinearities to the ARM (violates full hardware path); adopting bfloat16 wholesale solely because TATAA does so; copying DSP48E2 packing numbers to DSP48E1; replacing LN with BN/ReLU and presenting it as unchanged-model optimization; generic power/endurance/clock sweeps without an architectural comparison; building larger models to win software accuracy; sparse/linear attention without an explicit approved new-model research branch. These can be optional follow-ups, but none is the best first use of the available board.

**Publication honesty.** A single exact narrowing that saves a handful of DSPs is probably insufficient as a standalone strong method paper. A justified precision allocation with a reproducible failure mechanism, comparisons against signed/unsigned and logarithmic probability coding, actual post-route Pareto gains, and full system correctness is substantially stronger. Even that is an applied hardware/system contribution whose venue strength depends on measured effect size and differentiation; no acceptance or novelty guarantee is warranted.

### Gaps

- We have not established that an existing published accelerator can be reproduced on this exact toolchain/device within reasonable effort. Cross-paper GOPS comparisons alone are weak because precision, model, device, resources and inference boundaries differ.
- If all numerical/physical gains are small, the best decision may be to write a rigorous engineering paper from the existing system rather than delay indefinitely in search of a headline trick.
