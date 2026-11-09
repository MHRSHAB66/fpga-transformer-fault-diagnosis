# K2 — SSH stream lost, workload completed on the board (2026-10-24)

- The PC stream session ended at 388 s with `client_loop: send disconnect: Connection reset`
  (`workload_stream.out`, `pc_rx.tsv`). run.json therefore records result FAILED/ABORTED with
  workload_rc/guard_rc = None (run.json is kept unchanged as written by the tool).
- Board log (`ssh_reset_board_logs.txt`): `macb e000b000.ethernet eth0: link down` at uptime
  1268.8 s (≈04:51:18 board clock), `link up (1000/Full)` 7 s later. No Windows System-log event
  for the PC adapter in that minute. The same board-side link-down was recorded on the stock
  kernel on 2026-10-23 (`artifacts/results/fpga_timing_20261023/diag_dma_ram_c_extload/OUTAGE_EVIDENCE.txt`),
  so it is not attributed to PREEMPT_RT; root cause unknown.
- The workload was NOT aborted and NOT retried: it kept running under its independent guard
  (nohup) until 04:55:05; guard.log ends with `# guard END ... (workload exited)`, no
  EMERGENCY_STOP. `remote/out/summary.json`: executed 30,200, aborted null, 0 misses.
  The remote dir was copied afterwards with scp (`remote/`).
- In stream mode the wrapper's WORKLOAD_RC/GUARD_RC lines go only to the (lost) SSH channel, so
  the exit codes are not recorded; the verifier's formal pass flag is therefore false for K2 even
  though every timing/class/logit condition holds on the raw npz.
