<#
  run_fpga_viewer.ps1 -- launch the FPGA fault-classification viewer (Replay + Live).

  Replay Mode needs only Python 3 + Tkinter (standard library). Live FPGA Mode
  additionally needs the OpenSSH client (ssh/scp) and a reachable board.

  Examples:
    .\run_fpga_viewer.ps1                 # both tabs
    .\run_fpga_viewer.ps1 -Mode replay    # recorded full-test replay only
    .\run_fpga_viewer.ps1 -Mode live      # real on-board inference only
    .\run_fpga_viewer.ps1 -ResultDir artifacts\results\fpga_fulltest_20261008_141335
#>
param(
    [ValidateSet("both", "replay", "live")] [string]$Mode = "both",
    [string]$ResultDir = "",
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$argv = @("-m", "host.fpga_result_viewer", "--mode", $Mode)
if ($ResultDir -ne "") { $argv += @("--result-dir", $ResultDir) }

Write-Host "Launching: $Python $($argv -join ' ')" -ForegroundColor Cyan
& $Python @argv
