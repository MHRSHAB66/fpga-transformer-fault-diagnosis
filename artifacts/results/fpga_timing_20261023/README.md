# Tail-latency study on the physical board (2026-10-23)

Deployment: `/home/xilinx/deployments/seq256_argmax_review_d55cbdfe469f` (BIT sha256
`ce3c254b…`, HWH `eabc0163…`). Every run below was launched by
`tools/run_board_guarded.py`: recorded read-only preflight (`preflight.json`), a new
remote dir `/home/xilinx/timing_runs_20261023/<tag>`, the independent guard
(`temp_guard.sh`, stop at 75 °C), exit codes and a post-run process check
(`run.json`). Nothing was retried automatically. Report:
`docs/reports/gui_demo_timing_20261023.md`. Summary table: `timing_runs_summary.json`.

| dir | what | setup |
|---|---|---|
| `smoke_diag_read`, `diag_read_full` | gzip read only, no FPGA (`tools/board/fpga_tail_diag.py read`) | per-window wall/CPU time + getrusage |
| `diag_dma_ram_b` | DMA + FPGA only, 256 windows cycled from RAM, 20,000 inferences | no file I/O in the loop |
| `diag_dma_ram_c_extload` | same, under external SSH load — **interrupted**: Ethernet link loss + board power cycle at ~2 min; no result (`OUTAGE_EVIDENCE.txt`) | not retried |
| `diag_dma_ram_NOT_RUN_setsid_f_unsupported` | launch failed (`setsid -f` missing on the board); workload never started | — |
| `A_legacy_unpaced_full` | runner, original overlap loop, 95,404 windows | output + predictions written to the board's SD card; wrapper polled over SSH every 10 s |
| `B_prefetch128_unpaced_full` | runner, reader THREAD + 128-window queue | same setup as A (comparable to A) |
| `C_prefetch128_paced48828_full` | thread reader, windows released at 48.828 Hz | external SSH polling started during this run (22/min) — not comparable to A/B |
| `B2_proc128_unpaced_full` | reader PROCESS + 128-window queue | external SSH polling ~36/min — not comparable to A/B |
| `S1…S4_*_stream_30k` | live path: runner output streamed over one SSH session to the PC; L = original loop, RA32 = + 32 MB kernel readahead; alternating | 30,000 contiguous windows each |
| `board_unittest_*` | `tests/test_runner_prefetch.py` executed on the board (fork available), no FPGA | — |
| `argmax_fulltest_logits_check.json` | logits/CONF of the 2026-10-23 argmax full test vs Python INT10 reference | 95,404 windows |

`timing.npz`-style raw files are the `workload.out` / `workload_stream.out` /
`pc_rx.tsv` text files; `*.log` (guard logs) are matched by `.gitignore`.
The board clock is not synchronized (shows 2019); only intervals are meaningful.
