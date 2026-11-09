# Same hard-RT workload as tools/run_hard_rt_20261024.ps1, on the one-shot-booted
# PREEMPT_RT kernel (criterion: docs/reports/hard_rt_preempt_rt_criterion_20261024.md).
# Refuses to start a run unless `uname -v` reports PREEMPT RT. No retry.
param([string[]]$Only)
Set-Location (Split-Path -Parent $PSScriptRoot)
$base = "artifacts/results/hard_rt_preempt_rt_20261024"
New-Item -ItemType Directory -Force $base | Out-Null
$runs = @(
  @{tag="K1_rtkernel_fifo50_cpu1"; n="30200"},
  @{tag="K2_rtkernel_fifo50_cpu1_repeat"; n="30200"},
  @{tag="K3_rtkernel_fifo50_cpu1_100k"; n="100200"}
)
foreach ($r in $runs) {
  if ($Only -and ($Only -notcontains $r.tag)) { continue }
  $k = ssh -i "$HOME\.ssh\id_zturn" -o BatchMode=yes xilinx@192.168.2.99 "uname -a; cat /sys/kernel/realtime"
  $k | Out-File -Encoding utf8 "$base/$($r.tag)_kernel.txt"
  if (-not (($k -join ' ') -match 'PREEMPT RT')) { Write-Output "STOP: kernel is not PREEMPT_RT"; break }
  python -m tools.run_board_guarded --stream --tag "rtk20261024_$($r.tag)" --out "$base/$($r.tag)" --files tools/board/fpga_rt_acceptance.py --max-minutes 60 -- python3.6 -u "{DIR}/fpga_rt_acceptance.py" "{BIT}" "{IN}" 4096 $r.n 200 1 1 50 "{DIR}/out"
  if ($LASTEXITCODE -ne 0) { Write-Output "STOP: $($r.tag) did not finish OK"; break }
}
