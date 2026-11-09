# Hard real-time acceptance runs (criterion: docs/reports/hard_rt_acceptance_criterion_20261024.md).
# Stream mode: ONE SSH session, no polling. Each run: own preflight + new remote dir +
# independent guard (75 C). A run that does not finish OK stops the sequence (no retry).
param([string[]]$Only)
Set-Location (Split-Path -Parent $PSScriptRoot)
$base = "artifacts/results/hard_rt_20261024"
$runs = @(
  @{tag="R0_normal_prio"; rt="0"; cpu="1"; prio="0"},
  @{tag="R1_fifo50_cpu1"; rt="1"; cpu="1"; prio="50"},
  @{tag="R2_fifo50_cpu1_repeat"; rt="1"; cpu="1"; prio="50"},
  @{tag="L1_fifo50_cpu1_100k"; rt="1"; cpu="1"; prio="50"; n="100200"}
)
foreach ($r in $runs) {
  if ($Only -and ($Only -notcontains $r.tag)) { continue }
  python -m tools.run_board_guarded --stream --tag "rt20261024_$($r.tag)" --out "$base/$($r.tag)" --files tools/board/fpga_rt_acceptance.py --max-minutes 60 -- python3.6 -u "{DIR}/fpga_rt_acceptance.py" "{BIT}" "{IN}" 4096 $(if ($r.n) { $r.n } else { "30200" }) 200 $r.rt $r.cpu $r.prio "{DIR}/out"
  if ($LASTEXITCODE -ne 0) { Write-Output "STOP: $($r.tag) did not finish OK"; break }
}
