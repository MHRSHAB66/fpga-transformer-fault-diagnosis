# Live guard and argmax build — 2026-10-22

The live client creates a fresh `/home/xilinx/live_runs/<id>` directory for each
run and requires the read-only preflight and a separate temperature guard. The
guard terminates the workload on a reading at or above 75 °C, an invalid raw
reading, or invalid calibration. The SSH wrapper also terminates the workload
if the guard exits unexpectedly. No run is retried automatically.

Physical smoke test of the existing deployed overlay: four windows, four
matches with the recorded board predictions, guard exit 0, temperature
50.078 °C. Evidence: `artifacts/results/fpga_live_test_20261022_151348/`.
This small single-class run is not a multiclass or GUI validation. Guard fault
injection used fake XADC files and a `sleep` process, not FPGA inference or
actual heating: invalid reading and 75.000 °C each produced exit 9; early guard
exit produced an abort. The board clock is not synchronized; remote log dates
do not represent the host date.

The new argmax HLS top passed Vitis 2020.2 C simulation: 40/40 exact vectors,
including 16 ties (`seq256_ingress1024_dsp_margin_argmax/sol1/csim/report/transformer_top_csim.log`).
HLS C synthesis estimated interval 1,490,531 cycles, 145 BRAM18K, 156 DSP,
47,003 FF, 53,041 LUT (`.../sol1/syn/report/transformer_top_csynth.rpt`).
These are estimates, not post-route or physical measurements. Vivado 2020.2 IP
export initially failed because its generated revision exceeded its integer
range. The generated packaging Tcl was preserved as
`run_ippack_original_20261022.tcl`; its new-build copy was repackaged with
revision 1. The installed tool was not modified.

The new system was implemented and tested on the physical board. Post-route
timing reports WNS +0.250 ns and WHS +0.010 ns; utilization is 31,766 LUT,
30,952 registers, 70.5 BRAM tiles and 210 DSP. See
`artifacts/results/seq256_dma_argmax_z7020_20261022/vivado_reports/` for the
complete reports and warnings. On the board, 40/40 golden vectors, including
16 ties, matched both class and logits; class came from the FPGA register.
See `artifacts/results/seq256_dma_argmax_z7020_20261022/physical_golden_output.txt`.

A separate balanced live run executed 40 windows: 39/40 correct against ground
truth and 40/40 matching the previous recorded board predictions. Its maximum
reported temperature was 50.078 C. Evidence:
`artifacts/results/fpga_live_test_20261022_160622/`. Its reported local BIT/HWH
hashes mistakenly refer to the earlier bundle; the deployed package's actual
local BIT SHA-256 is `ce3c254b86797c58bb99e7200169db520ff39eddfd3e2a6c559b16b5846f166b`
and HWH SHA-256 is `eabc0163be6ae333f0d32fec391708fa8bd8a097854cf7b73cf9d8387ba9a19c`.
The package manifest was checked on the board before the run. The logging code
has been corrected; historical evidence has not been overwritten.

On 2026-10-23 the new overlay also completed all 95,404 sequential test
windows, with 93,275 correct (97.768437%), and 95,404/95,404 class predictions
matching the earlier overlay. The independent guard exited 0; the reported peak
temperature was 49.708 C. The independent recheck verified all dataset indices
in order. Mean recorded inference-loop latency was 16.092 ms; 143 windows
exceeded 20.48 ms, maximum 664.945 ms. The measured 60.851 windows/s is an
average for this prepared-input run, not a hard-deadline guarantee. Evidence:
`artifacts/results/fpga_argmax_fulltest_20261023/live_evidence.json` and
`live_frames.jsonl`. The GUI Start/Stop path has not been physically validated
with this deployment.
