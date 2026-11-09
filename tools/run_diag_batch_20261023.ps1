# Diagnostic batch (2026-10-23): DMA-only repeat under the current external SSH
# load, then alternating read-only readahead tests (no FPGA in the read runs).
# Each run: own preflight + new remote dir + independent guard. Stops on failure.
Set-Location (Split-Path -Parent $PSScriptRoot)
$base = "artifacts/results/fpga_timing_20261023"
$runs = @(
  @{tag="diag_dma_ram_c_extload"; args=@("dma","{BIT}","{IN}","256","20000","{DIR}/out")},
  @{tag="diag_read_ra0_a"; args=@("read","{IN}","95404","{DIR}/out","0")},
  @{tag="diag_read_ra32_a"; args=@("read","{IN}","95404","{DIR}/out","32")},
  @{tag="diag_read_ra0_b"; args=@("read","{IN}","95404","{DIR}/out","0")},
  @{tag="diag_read_ra32_b"; args=@("read","{IN}","95404","{DIR}/out","32")}
)
foreach ($r in $runs) {
  python -m tools.run_board_guarded --tag $r.tag --out "$base/$($r.tag)" --files tools/board/fpga_tail_diag.py --max-minutes 15 -- python3.6 -u "{DIR}/fpga_tail_diag.py" @($r.args)
  if ($LASTEXITCODE -ne 0) { Write-Output "STOP: $($r.tag) did not finish OK"; break }
}
