#!/bin/bash
cd /home/xilinx/deployments/seq256_argmax_review_d55cbdfe469f/driver || { echo 'ABORT no driver dir'; echo DONE; exit 1; }
export PYTHONPATH=/home/xilinx/deployments/seq256_argmax_review_d55cbdfe469f/driver
date -u +START_UTC=%FT%TZ
python3.6 -u /home/xilinx/timing_runs_20261023/rt20261024_R2_fifo50_cpu1_repeat/fpga_rt_acceptance.py /home/xilinx/deployments/seq256_argmax_review_d55cbdfe469f/overlay/design_1_wrapper.bit /home/xilinx/fpga_fulltest_20261008/fpga_testset_in.i32.gz 4096 30200 200 1 1 50 /home/xilinx/timing_runs_20261023/rt20261024_R2_fifo50_cpu1_repeat/out 2>&1 & P=$!
bash /home/xilinx/timing_runs_20261023/rt20261024_R2_fifo50_cpu1_repeat/temp_guard.sh $P /home/xilinx/timing_runs_20261023/rt20261024_R2_fifo50_cpu1_repeat/guard.log 75 1 & G=$!
while kill -0 $P 2>/dev/null; do
  if ! kill -0 $G 2>/dev/null; then kill -TERM $P 2>/dev/null; sleep 2; kill -KILL $P 2>/dev/null;
    echo 'ABORT independent guard exited while workload alive'; fi
  sleep 1
done
wait $P; R=$?; wait $G; S=$?
echo WORKLOAD_RC=$R; echo GUARD_RC=$S; date -u +END_UTC=%FT%TZ; echo DONE
