#!/usr/bin/env python3
"""Bounded on-board comparison of the new class register and golden logits.

Run only under a separate temperature guard. This script also requires a fresh
XADC reading before loading the overlay and checks temperature before each DMA.
"""
import os
import signal
import sys
import time

import numpy as np

from fault_classifier import FaultClassifier, HardwareBackend
from fpga_live_runner import TempMonitor, STOP_C, TEMP_STALE_S


def main():
    bit, directory = sys.argv[1:3]
    monitor = TempMonitor(interval=1.0)
    monitor.start()
    start = time.monotonic()
    temp, stale = monitor.snapshot()
    while temp is None and time.monotonic() - start < TEMP_STALE_S:
        time.sleep(0.5)
        temp, stale = monitor.snapshot()
    if temp is None or temp >= STOP_C or stale is None or stale >= TEMP_STALE_S:
        raise RuntimeError('temperature preflight failed')
    print('TEMP_START=%.3f' % temp, flush=True)
    backend = HardwareBackend(bit, wait_timeout=10.0)
    classifier = FaultClassifier(backend=backend)
    if classifier.class_source != 'fpga_register':
        raise RuntimeError('hardware class register not present')
    if backend.dma.buffer_max_size < 32768:
        raise RuntimeError('DMA input transfer cannot fit one complete frame')
    with open(os.path.join(directory, 'golden_in.txt')) as inputs, \
            open(os.path.join(directory, 'golden_expected.txt')) as expected:
        count, seq, channels = map(int, inputs.readline().split())
        # 40 vectors for golden_ingress1024_argmax, 41 for golden_raw1024_argmax
        if count < 1 or (seq, channels) != (1024, 8):
            raise RuntimeError('unexpected golden geometry')
        expected.readline()
        ties = 0
        for i in range(count):
            temp, stale = monitor.snapshot()
            if temp is None or temp >= STOP_C or stale is None or stale >= TEMP_STALE_S:
                raise RuntimeError('temperature safety stop before vector %d' % i)
            window = np.fromstring(inputs.readline(), dtype=np.int32, sep=' ')
            if window.size != seq * channels:
                raise RuntimeError('invalid input at vector %d' % i)
            row = [int(x) for x in expected.readline().split()]
            if len(row) != 7 or row[0] != i:
                raise RuntimeError('invalid expected output at vector %d' % i)
            signal.alarm(20)
            try:
                cls, logits = classifier.infer_from_codes(window.reshape(seq, channels))
            finally:
                signal.alarm(0)
            if cls != row[2] or list(logits) != row[3:]:
                raise RuntimeError('mismatch at vector %d: class=%s logits=%s' % (i, cls, logits))
            ties += int(row[3:].count(max(row[3:])) > 1)
            print('VECTOR %d class=%d logits=%s temp_c=%.3f' %
                  (i, cls, ','.join(str(int(x)) for x in logits), temp), flush=True)
    monitor.stop()
    print('PASS exact=%d/%d ties=%d class_source=%s' %
          (count, count, ties, classifier.class_source), flush=True)


if __name__ == '__main__':
    main()
