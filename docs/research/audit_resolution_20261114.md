# Resolution of the 2026-11-13 execution audit

Audit: `reports/claude_execution_review_20261113.md`, `artifacts/review/claude_execution_audit_20261113/`
(baseline commit `99617bc`; B04 was uncommitted work in progress). Every finding was re-checked on
the current tree before changing anything. Old results are kept unchanged; corrected experiments
live in new versioned directories and name what they supersede. Status words: **reproduced** (the
finding is real), **fixed now**, **already fixed**, **pending external dependency**.

## Findings

### 1. P1 — T01 short events: invalid A→B→A construction, different events per policy, late detections counted

- Status: reproduced; **fixed now** (T01 v2).
- Root cause: `tools/research/temporal_policies.py:111` picked a start index of class A without
  checking that the next 60 windows stay class A or inside one recording; one shared RNG was consumed
  sequentially per policy (different events per policy); the success window extended 20 windows past
  the event.
- Changed files: new `tools/research/temporal_policies_v2.py` (old tool untouched).
- Fix: one immutable event manifest per split, written before any policy runs
  (`artifacts/research/T01v2_20261114/event_manifest_*.json`, seed + sha256; recording = `index // 244`
  for the raw exports, `file_of` for the validation cache; every window label asserted); all policies
  run on the same events from the same fresh state; scoring separates detection during B, detection
  only after B (late), no detection, false switches outside the event, recovery to A and its delay.
- Validation: `python -m tools.research.temporal_policies_v2 selfcheck --out artifacts/research/T01v2_20261114`
  → 12/12 checks pass (recording-boundary violation detected, identical event ids across policies,
  zero-margin ties, events shorter than the required evidence, late detections, unresolved/timeout).
- Evidence: `artifacts/research/T01v2_20261114/summary.json`, `selfcheck.json`.
- Corrected result: on validation (selection) the evidence policy `evid_64` has the same step-change
  p95 delay as `vote_5` (2 windows, 400 events, 0 unresolved at H_MAX = 16 windows), so the frozen
  T01 gate (≥ 20 % lower p95 at matched false-switch budget and horizon) is **not passed** → FPGA
  implementation of T01 is **stopped** under this protocol. Secondary, non-gated observations kept:
  macro-F1 +0.40 pp vs vote_5 on validation; 2-window events detected during the event 262/300
  (vote_5 25/300, hyst_3 3/300); 7 vs 44 false switches before the event (400 step events).
- Claims to correct: ledger T01 ("gate formally passed"), claims table row 9, any "89 % vs 17 %"
  short-event statement (corrected test numbers, previously explored split: evid_64 266/300 during,
  vote_5 34/300 during + 13 only after the event).

### 2. P1 — Q01 calibrated confidence ignored `s_logits`

- Status: reproduced; **fixed now** (Q01 v2).
- Root cause: `tools/research/q01_quality.py:108` used `softmax(logits_int / T)`; the frozen contract
  `artifacts/calibration/conf_temperature_v1/temperature.json` is `max softmax(s_logits·logits_int / T)`.
- Changed files: new `tools/research/q01_quality_v2.py` (reads `T` and `s_logits` from the file;
  old tool untouched; the superseded formula is kept as a labelled comparator).
- Validation: same 2,048 + 2,048 windows and channels as Q01 (`sample_manifest.json`; perturbations
  asserted identical to Q01 for the original severities); thresholds from clean validation with the
  frozen rule.
- Changed conclusion: with the correct formula, calibrated confidence reaches 89.99 % clean validation
  coverage (just below the 90 % gate: ties at the 10th percentile) with −80.2 % accepted wrong
  outputs; the superseded formula had saturated (threshold 0.99999999) and behaved like the margin
  (90.4 %, −78.0 %). Conclusions about margin and flags are unchanged.

### 3. P1 — Research bundle metadata/README did not describe the real contract (A06 and the other raw bundles)

- Status: reproduced (A06 `attn_bits: 10`, README with normalized INT10 input and 16 goldens; the same
  generic text in the A05/B02/B03 raw bundles); **fixed now**.
- Root cause: `tools/package_seq256_pynq_deploy.py` always wrote the canonical ingress README and copied
  the canonical manifest/config.
- Changed files: `tools/research/variant_contracts.py` (one contract per research bundle:
  precision/signedness/qmax per stage, context requant for A06, raw24 input + on-FPGA preprocessing,
  reference and how to reconstruct it, golden count/ties/sha256, source/build identity, physical
  evidence, timing); `tools/package_seq256_pynq_deploy.py --contract` (renders README,
  `config/model_manifest.json`, `config/integer_config_scales.json`, `config/variant_contract.json`
  from the contract and refuses golden files that differ from the contract); `pynq/fault_classifier.py`
  (the repository NumPy backend refuses a bundle whose contract says its numerics are not canonical).
- Validation: the four bundles regenerated from their contracts; overlay files (bit/hwh/xsa)
  byte-identical to before; `MANIFEST.sha256` regenerated; A06 now records W10/A10/P8u
  (unsigned, qmax 255, `M_l0_ctx_p8u_q40 = 4322686742`) and "NOT a W8/A8 implementation".
- Evidence: `artifacts/research/contracts/*.json`, `artifacts/deploy/research_*/README.md`.

### 4. P2 — T01 "finite horizon" was only a bounded state

- Status: reproduced (a zero-margin stream never switches); **fixed now**.
- Fix: the method is renamed *bounded-state evidence*; an explicit maximum decision horizon
  H_MAX = 16 windows applies to every policy (unresolved/timeout counted, and counted at H_MAX in the
  p95); a separate `evidto_T_h` policy adds a timeout fallback (switch after h consecutive raw windows
  of the new class) that guarantees resolution of any persistent change with a constant raw class.
  Self-check: `zero_margin_evid_never_switches`, `zero_margin_timeout_switches_after_h`.

### 5. P2 — Q01 availability cost not reported

- Status: reproduced; **fixed now** (Q01 v2).
- Fix: full risk/coverage per method and condition (accepted, rejected, accepted wrong, accepted
  false-healthy, correct outputs rejected, per-class coverage, explicit `zero_accepted`), a severity
  sweep of every fault, nuisance white noise (20/10 dB) that should be accepted, and ordered fault
  bursts inside recordings (`burst_manifest.json`, onset/duration/channel frozen, identical streams).
- Reported explicitly: flags and flags+margin accept **zero** windows under bits8 and hold16 (they
  refuse to classify there; not higher accuracy); flags+margin covers only 48.9 % of clean normal-class
  validation windows (margin threshold), the flags alone 100 %; flags never reject the white-noise
  conditions; flags react in the first faulty window of a burst and recover in the first clean
  window; lag faults are invisible to the flags.
- Gate re-assessed with the frozen thresholds (not changed after seeing results): flags, margin and
  flags+margin pass; correct confidence misses coverage by 0.01 pp.

### 6. P2 — B03 physical acceptance, timing margin, recovery, archive, ledger gaps, reproduction

| item | status | evidence |
|---|---|---|
| B03 clean full test on its own overlay | **fixed now**: 95,404/95,404 logits+class = canonical C-model, `integrity_status` 0 on all | `artifacts/results/raw_core_board_fulltest_20261113_231912/` |
| B03 normal-operation real time | **fixed now**: 0 misses / 30,000, max 17.72 ms, 30,200 outputs = canonical | `artifacts/results/raw_core_board_rt_20261113_235257/` |
| B03 WNS 0.000 | reported as "met with no reported margin" everywhere; failing default run kept | `artifacts/research/B03_20261113/b03_build/first_impl_timing_fail/` |
| B03 recovery / persistent faults | recovery measured on the B04i overlay (clean window after each injection, see B04); persistent and vote-state faults not tested on hardware (scope stated) | `artifacts/research/B04_20261114/recovery/` |
| Analyzer robustness | **fixed now**: `b03_campaign analyze` asserts complete row counts, consecutive row ids, every manifest field per board row, records seed/executable sha256/long-width model/selected reference; `manifest --seed`, `cmodel --exe`, `--recovery` | `tools/research/b03_campaign.py` |
| Seed record of B03 | manifest regenerated from seed 20261115 byte-identical → `manifest_meta.json` | `artifacts/research/B03_20261113/manifest_meta.json` |
| Untracked strategy helper | **fixed now**: tracked copy of the helper actually used `tools/vivado/rerun_impl_strategy.tcl` (identical to the preserved `rerun_impl_strategy_used.tcl`); the user's untracked file is left alone | |
| Shorthand commands | **fixed now**: `docs/research/reproduce.md` rewritten with exact commands | |
| A04, Q02, Z01 ledger entries | **fixed now** (A04 added, Q02 added, Z01 tracked) | `docs/research/experiment_ledger.json` |
| Board runs without deployment identity | **fixed now**: every new run writes `run_meta.json` (deployment, bitstream sha256, command, git HEAD) | `tools/raw_core_board_test.py` |

### Side notes of the audit

- Repeated classifier tail (B02/B03): HLS time-multiplexes both evaluations on one unit; wording
  corrected in the ledger, contracts and `docs/research/b04_integrity_mechanism.md` (transient upset
  of one evaluation only; no persistent/common-mode coverage).
- A02 "0 zero rows in every variant": corrected — uniform INT8 has a zero-row rate of 3.7e-6
  (170 windows with ≥ 1 zero row); P8-signed has 0, so the A02 conclusion is unchanged.
- A06 is W10/A10/P8u, not W8/A8; W8A8+P8u passed the A02 validation accuracy gate but was not built
  (conditional; see the ledger).
- F00 source-hash differences in `pynq/fault_classifier.py` and `tools/board/fpga_live_runner.py`:
  conditional research-register reads only (integrity, quality, NumPy-backend contract guard); the
  hardware path of the canonical overlays is unchanged (canonical goldens re-run on the board
  2026-11-14: `raw_core_board_golden_20261114_001545`, `_001609`, both 41/41).

## B04 (new) and Q02 (new)

See the ledger entries `B04` and `Q02` for results; mechanism and arithmetic proof of B04:
`docs/research/b04_integrity_mechanism.md`.
