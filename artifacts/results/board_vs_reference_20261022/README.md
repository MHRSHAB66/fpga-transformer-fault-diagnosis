# Board vs reference — same input, per window (2026-10-22)

All sources evaluated on `artifacts/results/fpga_fulltest_20261008_141335/fpga_testset_in.i32.gz`
(95,404 windows). Nothing in the 2026-10-08 evidence was modified.

| path | content | kind |
|---|---|---|
| `csim_dsp_margin/csim_logits.txt` | g++ C-model of the deployed HLS top, `<idx> l0 l1 l2 l3` | MEASURED-SOFTWARE |
| `csim_argmax/csim_logits.txt` | g++ C-model of the argmax variant, 6th column = `pred_class` register | MEASURED-SOFTWARE |
| `numpy_reference/numpy_export_*` | Python INT10 reference, export (deployed) scales | MEASURED-SOFTWARE |
| `numpy_reference/numpy_calib12_*` | Python INT10 reference, 12-file calibration (old "software reference") | MEASURED-SOFTWARE |
| `comparison.json` | all agreements/metrics, incl. physical board logits (2026-10-22) | derived |
| `disagreements_board_vs_calib12.csv` | the 225 windows where board ≠ 12-file reference | derived |
| `confidence/` | softmax-confidence calibration (validation + test) | MEASURED-SOFTWARE |

Commands: `docs/reports/consolidated_results_20261022.md` §7.
