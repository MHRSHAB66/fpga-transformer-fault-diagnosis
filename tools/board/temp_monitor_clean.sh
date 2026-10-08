#!/bin/bash
# Independent XADC temperature monitor for the MYIR Z-Turn (Zynq-7020).
# Runs as its OWN process (not in the DMA/inference thread). Publishes the
# latest die temperature atomically to /tmp/zturn_temp_latest every 5 s and
# appends a timestamped line to the log. Stops when /tmp/zturn_temp_stop exists.
D=/sys/bus/iio/devices/iio:device0
LOG=/home/xilinx/temperature_monitor.log
STOP=/tmp/zturn_temp_stop
LATEST=/tmp/zturn_temp_latest
off=$(cat "$D/in_temp0_offset")
scl=$(cat "$D/in_temp0_scale")
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "# monitor START $(ts) pid=$$ offset=$off scale=$scl interval=5s" >> "$LOG"
while [ ! -f "$STOP" ]; do
  raw=$(cat "$D/in_temp0_raw" 2>/dev/null)
  if [ -z "$raw" ]; then
    echo "$(ts) INVALID raw_read_failed" >> "$LOG"
    echo "INVALID" > "$LATEST.tmp" && mv -f "$LATEST.tmp" "$LATEST"
  else
    t=$(awk -v r="$raw" -v o="$off" -v s="$scl" 'BEGIN{printf "%.3f",(r+o)*s/1000}')
    echo "$(ts) temp_c=$t raw=$raw" >> "$LOG"
    echo "$t" > "$LATEST.tmp" && mv -f "$LATEST.tmp" "$LATEST"
  fi
  sleep 5
done
echo "# monitor STOP $(ts)" >> "$LOG"
