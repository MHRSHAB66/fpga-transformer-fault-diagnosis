# Live-path before/after (2026-10-23): runner output streamed over ONE SSH session to
# the PC (as in the demo; no board-side log, no prediction file, no polling).
# Alternating L (legacy overlap loop, no readahead) / RA (same loop + 32 MB kernel
# readahead) so changing external conditions hit both variants. 30,000 contiguous
# windows each. Each run: own preflight + new remote dir + independent guard.
Set-Location (Split-Path -Parent $PSScriptRoot)
$base = "artifacts/results/fpga_timing_20261023"
$runs = @(
  @{tag="S1_L_stream_30k"; ra="0"}, @{tag="S2_RA32_stream_30k"; ra="32"},
  @{tag="S3_L_stream_30k"; ra="0"}, @{tag="S4_RA32_stream_30k"; ra="32"}
)
foreach ($r in $runs) {
  python -m tools.run_board_guarded --stream --tag $r.tag --out "$base/$($r.tag)" --files tools/board/fpga_live_runner.py -- FPGA_RUNNER_PREFETCH=0 "FPGA_RUNNER_READAHEAD_MB=$($r.ra)" FPGA_RUNNER_TIMING=1 python3.6 -u "{DIR}/fpga_live_runner.py" "{BIT}" "{IN}" 1024 8 30000 -
  if ($LASTEXITCODE -ne 0) { Write-Output "STOP: $($r.tag) did not finish OK"; break }
  python -m tools.analyze_runner_timing "$base/$($r.tag)/workload_stream.out" --pc-rx "$base/$($r.tag)/pc_rx.tsv" --out "$base/$($r.tag)/timing_summary.json"
}
