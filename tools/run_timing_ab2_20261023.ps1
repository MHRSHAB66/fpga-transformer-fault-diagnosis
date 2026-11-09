# Second sequence (process reader). Comparable before/after timing runs of the live runner on the board (2026-10-23).
# Each run: own preflight + new remote dir + independent guard (tools/run_board_guarded.py).
# A run that does not finish OK stops the sequence (no automatic retry).
Set-Location (Split-Path -Parent $PSScriptRoot)
$base = "artifacts/results/fpga_timing_20261023"
$runs = @(
  
  @{tag="B2_proc128_unpaced_full"; env="FPGA_RUNNER_PREFETCH=128 FPGA_RUNNER_PREFETCH_KIND=process FPGA_RUNNER_TIMING=1"},
  @{tag="C2_proc128_paced48828_full"; env="FPGA_RUNNER_PREFETCH=128 FPGA_RUNNER_PREFETCH_KIND=process FPGA_RUNNER_TIMING=1 FPGA_RUNNER_PACE_HZ=48.828125"}
)
foreach ($r in $runs) {
  $envs = $r.env.Split(" ")
  python -m tools.run_board_guarded --tag $r.tag --out "$base/$($r.tag)" --files tools/board/fpga_live_runner.py --max-minutes 45 -- @envs python3.6 -u "{DIR}/fpga_live_runner.py" "{BIT}" "{IN}" 1024 8 95404 "{DIR}/pred.txt"
  if ($LASTEXITCODE -ne 0) { Write-Output "STOP: $($r.tag) did not finish OK"; break }
  python -m tools.analyze_runner_timing "$base/$($r.tag)/remote/workload.out" --out "$base/$($r.tag)/timing_summary.json"
}
