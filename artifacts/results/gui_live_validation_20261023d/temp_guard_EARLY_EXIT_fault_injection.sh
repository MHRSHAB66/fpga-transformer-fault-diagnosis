#!/bin/bash
# Independent die-temperature guard (separate process from the workload).
# Usage: temp_guard.sh <pid> <log> [stop_c=75] [interval_s=1]
# Reads the Zynq XADC every interval; if temp >= stop_c it sends SIGTERM then
# SIGKILL to <pid>, writes EMERGENCY_STOP to the log and exits. It also exits
# when <pid> ends. It never restarts anything.
PID="$1"; LOG="$2"; STOP_C="${3:-75}"; IV="${4:-1}"
D=${XADC_DIR:-/sys/bus/iio/devices/iio:device0}
off=$(cat "$D/in_temp0_offset" 2>/dev/null); scl=$(cat "$D/in_temp0_scale" 2>/dev/null)
stop_workload() {
  echo "EMERGENCY_STOP $1" >> "$LOG"
  kill -TERM "$PID" 2>/dev/null; sleep 2; kill -KILL "$PID" 2>/dev/null
  exit 9
}
valid_number() { echo "$1" | awk '/^-?[0-9]+([.][0-9]+)?$/ {ok=1} END {exit !ok}'; }
if ! valid_number "$off" || ! valid_number "$scl"; then
  stop_workload "invalid XADC calibration"
fi
echo "# guard START $(date -u +%FT%TZ) pid=$PID stop_c=$STOP_C interval=${IV}s" >> "$LOG"
N_EARLY=0
while kill -0 "$PID" 2>/dev/null; do
  N_EARLY=$((N_EARLY+1)); if [ $N_EARLY -gt 3 ]; then echo "FAULT_INJECTION guard exits early" >> "$LOG"; exit 0; fi
  raw=$(cat "$D/in_temp0_raw" 2>/dev/null)
  if valid_number "$raw"; then
    t=$(awk -v r="$raw" -v o="$off" -v s="$scl" 'BEGIN{printf "%.3f",(r+o)*s/1000}')
    echo "$(date -u +%FT%TZ) temp_c=$t" >> "$LOG"
    if awk -v t="$t" -v s="$STOP_C" 'BEGIN{exit !(t>=s)}'; then
      stop_workload "temp_c=$t >= $STOP_C"
    fi
  else
    stop_workload "invalid temperature reading"
  fi
  sleep "$IV"
done
echo "# guard END $(date -u +%FT%TZ) (workload exited)" >> "$LOG"
