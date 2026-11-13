# Build a raw-input core variant end to end on this PC (Vitis HLS + Vivado 2020.2):
#   1. Vitis HLS C-simulation (golden_raw1024_argmax), C-synthesis, IP export
#   2. re-package the IP with core revision "1" (Vitis HLS 2020.2 writes a
#      date-based revision > 2^31 after 2021 and the packaging fails; the same
#      fix was used for the argmax IP on 2026-10-22)
#   3. Vivado system build -> bitstream, post-route reports, XSA in tmp/<proj>/
# Usage (from the repository root):
#   powershell -File tools/build_raw_core.ps1               # raw-input core (default)
#   powershell -File tools/build_raw_core.ps1 -Variant vote # raw-input core + temporal vote
#   powershell -File tools/build_raw_core.ps1 -Variant votena # same, noise-augmented weights (paper study)
#   powershell -File tools/build_raw_core.ps1 -Variant divq9  # research A05: exact 9-bit softmax divider
#   powershell -File tools/build_raw_core.ps1 -Variant b02    # research B02: selective integrity checks
#   powershell -File tools/build_raw_core.ps1 -Variant b03    # research B03: B02 + programmable fault injection
#   powershell -File tools/build_raw_core.ps1 -Variant a06    # research A06: 8-bit unsigned probabilities + exact divider
#   powershell -File tools/build_raw_core.ps1 -Variant b04    # research B04: fast checked projection + row checksum
#   powershell -File tools/build_raw_core.ps1 -Variant b04i   # research B04i: B04 + programmable fault injection
#   powershell -File tools/build_raw_core.ps1 -Variant q02    # research Q02: input-quality / decision-status registers
param([ValidateSet("raw", "vote", "votena", "divq9", "b02", "b03", "a06", "b04", "b04i", "q02")][string]$Variant = "raw")
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$hls = "G:\Xilinx\Vitis_HLS\2020.2\bin\vitis_hls.bat"
$vivado = "G:\Xilinx\Vivado\2020.2\bin\vivado.bat"
$cfg = @{
    raw  = @{ hlsTcl = "run_seq256_raw1024_dsp_margin_argmax_verify_and_synth.tcl"; hlsProj = "seq256_raw1024_dsp_margin_argmax";
              sysTcl = "run_seq256_dma_system_dsp_margin_raw_argmax.tcl"; sysProj = "vraw"; logs = "raw_core_build" }
    vote = @{ hlsTcl = "run_seq256_rawvote1024_verify_and_synth.tcl"; hlsProj = "seq256_rawvote1024";
              sysTcl = "run_seq256_dma_system_dsp_margin_vote.tcl"; sysProj = "vvote"; logs = "vote_core_build" }
    # research A05: exact 9-bit-quotient softmax divider (hls/research/a04_softmax_div)
    divq9 = @{ hlsTcl = "..\..\research\a04_softmax_div\run_divq9_verify_and_synth.tcl"; hlsProj = "seq256_divq9";
               sysTcl = "..\..\research\a04_softmax_div\run_dma_system_divq9.tcl"; sysProj = "vdq9"; logs = "divq9_core_build" }
    # research B02: selective integrity checks (hls/research/b02_integrity)
    b02 = @{ hlsTcl = "..\..\research\b02_integrity\run_b02_verify_and_synth.tcl"; hlsProj = "seq256_b02";
             sysTcl = "..\..\research\b02_integrity\run_dma_system_b02.tcl"; sysProj = "vb02"; logs = "b02_core_build" }
    # research B03: B02 + programmable logical fault injection (hls/research/b03_inject)
    b03 = @{ hlsTcl = "..\..\research\b03_inject\run_b03_verify_and_synth.tcl"; hlsProj = "seq256_b03";
             sysTcl = "..\..\research\b03_inject\run_dma_system_b03.tcl"; sysProj = "vb03"; logs = "b03_core_build" }
    # research B04: single-pipeline checked projection + per-row checksum (hls/research/b04_integrity_fast)
    b04 = @{ hlsTcl = "..\..\research\b04_integrity_fast\run_b04_verify_and_synth.tcl"; hlsProj = "seq256_b04";
             sysTcl = "..\..\research\b04_integrity_fast\run_dma_system_b04.tcl"; sysProj = "vb04"; logs = "b04_core_build" }
    # research B04i: B04 + programmable logical fault injection (-DB04_INJECT)
    b04i = @{ hlsTcl = "..\..\research\b04_integrity_fast\run_b04i_verify_and_synth.tcl"; hlsProj = "seq256_b04i";
              sysTcl = "..\..\research\b04_integrity_fast\run_dma_system_b04i.tcl"; sysProj = "vb04i"; logs = "b04i_core_build" }
    # research Q02: input-quality / decision-status metadata registers (hls/research/q02_quality)
    q02 = @{ hlsTcl = "..\..\research\q02_quality\run_q02_verify_and_synth.tcl"; hlsProj = "seq256_q02";
             sysTcl = "..\..\research\q02_quality\run_dma_system_q02.tcl"; sysProj = "vq02"; logs = "q02_core_build" }
    # research A06: unsigned 8-bit attention probabilities + exact 8-bit divider (hls/research/a06_p8u)
    a06 = @{ hlsTcl = "..\..\research\a06_p8u\run_a06_verify_and_synth.tcl"; hlsProj = "seq256_p8u";
             sysTcl = "..\..\research\a06_p8u\run_dma_system_a06.tcl"; sysProj = "vp8u"; logs = "a06_core_build" }
    # paper study: noise-augmented weights on the vote core (deployed model untouched)
    votena = @{ hlsTcl = "run_seq256_rawvote1024_noiseaug_verify_and_synth.tcl"; hlsProj = "seq256_rawvote1024_na";
                sysTcl = "run_seq256_dma_system_dsp_margin_vote_noiseaug.tcl"; sysProj = "vna"; logs = "vote_na_core_build" }
}[$Variant]
$logs = Join-Path $repo ("tmp\" + $cfg.logs)
New-Item -ItemType Directory -Force $logs | Out-Null

Set-Location $repo
& $hls -f ("hls\f1_int10_seq256\vitis\" + $cfg.hlsTcl) *> (Join-Path $logs "hls.log")
$ipdir = Join-Path $repo ($cfg.hlsProj + "\sol1\impl\ip")
$pack = Join-Path $ipdir "run_ippack.tcl"
(Get-Content $pack) -replace '^set Revision\s+"\d+"', 'set Revision    "1"' | Set-Content -Encoding ascii $pack
Set-Location $ipdir
& $vivado -notrace -mode batch -source run_ippack.tcl *> (Join-Path $logs "ippack.log")
if (-not (Test-Path (Join-Path $ipdir "xilinx_com_hls_transformer_top_1_0.zip"))) { throw "IP packaging failed" }

Remove-Item -Recurse -Force (Join-Path $repo ("tmp\" + $cfg.sysProj)) -ErrorAction SilentlyContinue
Set-Location $logs
& $vivado -mode batch -notrace -source (Join-Path $repo ("hls\f1_int10_seq256\vitis\" + $cfg.sysTcl)) *> (Join-Path $logs "vivado.log")
Set-Location $repo
"BUILD_DONE"
