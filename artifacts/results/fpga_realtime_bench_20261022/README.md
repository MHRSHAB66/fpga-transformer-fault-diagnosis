# Real-time measurements on the physical board (2026-10-22)

Board: MYIR Z-Turn XC7Z020, deployment `seq256_z7020_pynq_dma16_py36_20261008_131446`
(bitstream sha256 `43e43370…`, unchanged; not reflashed, no reboot). All new
board-side files live in `/home/xilinx/bench_20261022/`; no existing board file was
overwritten. Every run had an independent temperature-guard process (stop at 75 °C,
`tools/board/temp_guard.sh`) plus the in-process monitor; no stop occurred
(max 51.9 °C).

| path | run |
|---|---|
| `board_smoke_*`, `board_full_*` | `tools/board/fpga_stage_bench.py` (serial = the 2026-10-08 loop; pipelined = overlapped read), 300 / 95,404 windows, launched by `bench_run.sh` |
| `e2e/<tag>/` | `tools/measure_live_e2e.py`: live runner → SSH → PC `parse_line`; runner versions archived here (`fpga_live_runner_HEAD_e42ec4f.py`, `…_20261022a.py`, `…_20261022b.py` = repository) |
| `realtime_verdict.json` | `tools/summarize_realtime_20261022.py` (reads the files above) |

`timing.npz` and `guard.log` files are matched by `.gitignore`; they are needed to
regenerate `realtime_verdict.json` and to show the temperature record.
