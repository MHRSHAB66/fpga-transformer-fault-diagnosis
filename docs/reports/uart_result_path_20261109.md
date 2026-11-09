# Result transfer over the board's USB-UART — physical test (2026-11-09)

The proposal's block diagram names a UART/USB link from the FPGA system to the
PC. Until now the UART (MYIR Z-Turn on-board CP2102, COM6, 115200 8N1) carried
only the Linux boot console and the results travelled over Ethernet. This test
sends the classification of every window over that UART.

## Path

FPGA class (`pred_class` register of the deployed argmax overlay) → Zynq PS UART
(`/dev/ttyPS0`, raw 115200 8N1) → CP2102 USB-UART → PC COM6 → production parser
(`host/frame_protocol.py`), one text frame per window:

    FRAME=<k>,CLASS=<id>,LABEL=<name>,CONF=<0..1>

CONF is display-only: softmax of the FPGA's 4 logits with the deployed logits
scale and the validation-fitted temperature T = 1.4248, computed on the ARM; it
never changes the class. During the run the serial console (getty) is stopped and
the kernel console level lowered so the link carries only frames; both are
restored afterwards (no reboot, no flash).

- Board sender: `tools/board/fpga_uart_demo.py` (windows of the 256-window
  class-balanced RAM set, released every 20.48 ms, under the temperature guard).
- PC side and check: `tools/uart_result_test.py` (control over SSH, data only over
  the UART; compares every received frame with the frame sent, with the reference
  logits of that dataset window and with the PC-side calibrated CONF).

## Results

| Run | Frames sent | Received, parsed | Malformed / gaps / duplicates | Received = sent (bytes) | Class + logits = reference | CONF = PC value | Evidence |
|---|---|---|---|---|---|---|---|
| smoke, `tcdrain` per frame | 256 | 256 | 0 / 0 / 0 | 256 | 256 | 256 | `artifacts/results/uart_results_20261109_205221/` |
| smoke, final sender | 256 | 256 | 0 / 0 / 0 | 256 | 256 | 256 | `…_205328/` |
| **stability, final sender** | **10,000** | **10,000** | **0 / 0 / 0** | **10,000** | **10,000** | **10,000** | `…_205416/` |

Stability run: 10,000 frames in 204.9 s on the PC = **48.80 frames/s**, i.e. the
real-time window rate (50 kHz / 1,024 = 48.83 /s); board inference p50 15.95 ms,
p99 18.59 ms; UART write 0.48 ms mean; die temperature 54 °C; console restored.

## Findings and limits

- The first sender waited for each frame to leave the UART (`tcdrain`). On this
  kernel that wait costs ~24 ms (10 ms jiffy polling), longer than the 20.48 ms
  period, so the sender fell behind. The final sender only queues the frame
  (0.48 ms) and drains once at the end; 52 bytes per 20.48 ms use ~23 % of
  115200 baud.
- The board loop runs with default scheduling; its releases are paced but not
  hard real-time (4,112 of 10,000 iterations started after their release time
  following short stalls, while the average rate stayed at 48.80 /s). Hard
  real-time is verified only for the board-side RAM path (hard_rt reports).
- PC receive timestamps come in batches (USB-serial driver and 0.2 s read
  timeout), so they show throughput, not per-frame latency.
