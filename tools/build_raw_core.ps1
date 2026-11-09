# Build the raw-input core end to end on this PC (Vitis HLS + Vivado 2020.2):
#   1. Vitis HLS C-simulation (golden_raw1024_argmax), C-synthesis, IP export
#   2. re-package the IP with core revision "1" (Vitis HLS 2020.2 writes a
#      date-based revision > 2^31 after 2021 and the packaging fails; the same
#      fix was used for the argmax IP on 2026-10-22)
#   3. Vivado system build -> bitstream, post-route reports, XSA in tmp/vraw/
# Usage (from the repository root):  powershell -File tools/build_raw_core.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$hls = "G:\Xilinx\Vitis_HLS\2020.2\bin\vitis_hls.bat"
$vivado = "G:\Xilinx\Vivado\2020.2\bin\vivado.bat"
$logs = Join-Path $repo "tmp\raw_core_build"
New-Item -ItemType Directory -Force $logs | Out-Null

Set-Location $repo
& $hls -f "hls\f1_int10_seq256\vitis\run_seq256_raw1024_dsp_margin_argmax_verify_and_synth.tcl" *> (Join-Path $logs "hls.log")
$ipdir = Join-Path $repo "seq256_raw1024_dsp_margin_argmax\sol1\impl\ip"
$pack = Join-Path $ipdir "run_ippack.tcl"
(Get-Content $pack) -replace '^set Revision\s+"\d+"', 'set Revision    "1"' | Set-Content -Encoding ascii $pack
Set-Location $ipdir
& $vivado -notrace -mode batch -source run_ippack.tcl *> (Join-Path $logs "ippack.log")
if (-not (Test-Path (Join-Path $ipdir "xilinx_com_hls_transformer_top_1_0.zip"))) { throw "IP packaging failed" }

Remove-Item -Recurse -Force (Join-Path $repo "tmp\vraw") -ErrorAction SilentlyContinue
Set-Location $logs
& $vivado -mode batch -notrace -source (Join-Path $repo "hls\f1_int10_seq256\vitis\run_seq256_dma_system_dsp_margin_raw_argmax.tcl") *> (Join-Path $logs "vivado.log")
Set-Location $repo
"BUILD_DONE"
