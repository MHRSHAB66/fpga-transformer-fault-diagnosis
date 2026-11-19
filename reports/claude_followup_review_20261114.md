# Independent follow-up review of Claude's research work — 2026-11-14

Reviewed commit: `15676ff137bd33cd356e0661bd70081a3c42179c`.
Verdict: the principal corrections and new physical results are supported by the saved evidence. Several reporting and verification defects remain; this is not an unconditional approval of every output or completion of every research branch.

This review used local source, raw board logs, timing traces, deployment hashes and the actual backup. It did not access/reprogram the board, run Vivado, or change production code/results. Offline audit: `artifacts/review/claude_followup_audit_20261114/audit.py`; results: `artifacts/review/claude_followup_audit_20261114/results.json`.

## Confirmed from saved evidence

- The frozen model/export (5 files), canonical HLS sources (8), goldens (4) and reference outputs (2) are unchanged. All seven research bundles pass their own SHA256 manifests (13 files each). A06 metadata now correctly records W10/A10/P8u, unsigned probabilities, qmax 255, and its own reference.
- B03 and B04 each have 95,404 ordered clean board rows with exact canonical logits/class and zero integrity flags. Q02 has 95,404 exact clean rows and 3,840 exact replay rows, including all quality registers. The B04/Q02 recorded bitstream identities match the local bundles.
- Recomputed RT traces: B03 30,000 scored windows, zero misses, max 17.723684 ms; B04 zero/30,000, max 16.661509 ms; Q02 zero/30,000, max 16.725453 ms. Each has 30,200 exact outputs including warm-up. This is the finite RAM-release-to-result acceptance boundary, not acquisition/GUI WCET or an unlimited guarantee.
- B04i: all 1,408 rows match the LP64 reference on each of the old B03 and new confirmation manifests. There are respectively 93 and 80 class changes, all flagged; every one of the 1,152 injected faults in each campaign is flagged. Recovery probes contain 288 exact, unflagged clean windows after injected windows. Scope remains transient logical injection at three selected sites; a flag does not repair the class.
- B04/B04i/Q02 post-route utilization, timing and HLS latency summaries agree with the underlying reports. WNS is +0.024/+0.007/+0.147 ns respectively. The first two have very small timing margins.
- T01v2 recording boundaries, labels and manifest hashes pass. Event metrics recomputed for raw, vote_5, hyst_3 and evid_64 on all three streams agree with the saved summary (plus evidto_64_4 on validation). All 12 self-checks pass in a fresh review output directory. The validation gate fails (p95 2 versus 2 windows); stopping the FPGA branch is consistent with that result.
- The Q01 confidence formula now includes both frozen constants and returns 0.5577796419 for logits [20,10,0,-10]. Sample manifests reproduce. All 3,840 Q02 replay quality/status/margin values also match the independent Python feature reference when recomputed from the saved raw replay.
- A full fresh offline Q01v2 rerun completed: 2 splits × 17 conditions × 2,048 windows, plus 96 × 40 burst windows per split. Its complete summary, sample manifest and burst manifest equal the saved artifacts exactly. This reproduces the remaining recovery-reporting defect as well as the valid results; it is not an endorsement of that metric definition.
- Re-hashed all 1,050 backup files (7,818,269,728 bytes): no mismatches. Git bundle hash and `git bundle verify` pass. The snapshot's ledger predates the final backup-status commit; it is the only indexed source file changed since the snapshot. This is an identified snapshot difference, not backup corruption.

## Remaining findings

### P2 — Q02 replay accuracy uses the wrong labels

`tools/raw_core_board_test.py:150` always loads canonical test labels, including when `--input` specifies an ordered custom replay. Consequently `artifacts/results/raw_core_board_fulltest_20261114_214206/fulltest_check.json` reports accuracy **0.3825520833** (1,469/3,840). Recomputing against `artifacts/research/Q02_20261114/replay_labels.txt` gives **0.9666666667** (3,712/3,840). Logits/class/register equality remains correct; this defect concerns diagnostic accuracy reporting.

Add explicit custom-label support or omit accuracy when custom input has no established label manifest. Preserve the original result and save a corrected version with label identity. Do not use the 38.26% figure in the paper.

### P2 — Recovery counts include faults that never caused rejection

`tools/research/q01_quality_v2.py:199` counts the first accepted post-burst window as recovery even if no faulty window was rejected. `tools/research/q02_check.py:200` repeats that definition. Concrete example: test flags, `lag32/d20`, reports **detected 0/6, recovered 6/6, recovery delay 0**. That supports continued acceptance after the burst, not six successful recoveries from detected faults.

Separate never-rejected bursts from detected/rejected episodes; define eligibility and denominator for recovery and report unresolved cases. Qualify or replace the ledger sentence about recovery after every fault type. This does not invalidate the separate B04i injection/clean-pair recovery experiment.

### P2 — Verification commands can report success without validating their stated contract

`tools/package_seq256_pynq_deploy.py:161` does not validate the requested variant contract in `--check` mode. Reproduction during this audit: checking the A06 bundle with the **B04 contract** returns exit code 0 and `CHECK OK: bundle matches canonical sources`. The bundle itself is currently correct, but the verification command cannot establish contract agreement. Building research bundles without `--contract` can also restore generic metadata, so the research packaging path needs a guard or automatic contract selection.

Additionally `tools/research/q02_check.py` writes `exact_all`, decision-agreement booleans and mismatch counts without returning failure for numeric mismatches. Make these commands fail with a nonzero exit status on a failed acceptance check. Preserve their useful diagnostic output.

### P3 — C-model provenance is still missing in two primary B04i campaigns

Both `artifacts/research/B04_20261114/{heldout,b03_manifest}/summary.json` contain `cmodel_meta: null`; their directories lack `cmodel_meta.json`. The recovery directory has executable hashes and long-width identity. The two primary campaigns' saved board/reference rows match, but the requested executable provenance was not recorded there. Reconstruct and clearly label it, or rerun those small offline C-model campaigns with the hardened recorder. Do not retrospectively invent original-run metadata.

### P3 — The reproduction guide still omits required B04/B04i steps

`docs/research/reproduce.md:135` lists the default B04/B04i builds, then proceeds to campaign commands without the successful implementation-strategy rerun, build collection, contract packaging and deployment commands. The accepted artifacts required strategy retries; the default implementations failed timing. The ledger contains some of these details but still uses shorthand for B04i. Add complete, versioned commands for both accepted builds; the current guide is not a complete fresh-machine reproduction recipe. Its shell setup is Bash syntax despite the introductory PowerShell wording.

## Work that is explicitly not complete

- Z01's copy is on E:, a partition of the same physical SSD as G:. A copy on another physical device/cloud and destination verification remain pending.
- W8A8+P8u hardware, combined optimized/protected overlays and other conditional research branches have not all been implemented. Their conditional/deferred status must remain visible; A06 is not a W8A8 implementation.
- Power/energy, external acquisition/sensors, independent instrument timing and external-machine transfer require the listed equipment/data. Digital replay is not physical sensor validation.
- T01's FPGA implementation was stopped after a failed gate; this is a valid research outcome, not a completed positive contribution.

The remaining reporting fixes can mostly use existing raw evidence. No new full board campaign is justified solely by these findings. A final article still needs scoped claims and a coherent contribution; the existence of many completed experiments is not an acceptance guarantee.
