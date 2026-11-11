// Temporal majority vote over the last k window classes, inside the FPGA (2026-11-10).
//
// Bit-exact reference: host/temporal_vote.py. The core keeps the classes of the
// last VOTE_MAX windows (registers, initialised empty at bitstream load, never
// reset afterwards: a continuous stream). After each window it votes over the
// newest m = min(seen, k) entries, k = vote_k clamped to 1..VOTE_MAX; most votes
// wins, ties go to the class seen most recently. Outputs: winning class and its
// vote count. No DSP, no BRAM.
//
// Narrow types (2-bit class, 4-bit counts): the first build used 32-bit int for
// every value and cost ~3.4 k LUT + 2.3 k FF in the HLS estimate (+7.3 k LUT
// after Vivado synthesis), which congested routing on the 85 %-full device.
#ifndef TEMPORAL_VOTE_H
#define TEMPORAL_VOTE_H

#define VOTE_MAX 15

#if defined(__SYNTHESIS__) || defined(HLS_USE_AP_INT)
#include <ap_int.h>
  typedef ap_uint<2> vote_cls_t;   // class id 0..3
  typedef ap_uint<4> vote_cnt_t;   // 0..VOTE_MAX
#else
  typedef unsigned vote_cls_t;
  typedef unsigned vote_cnt_t;
#endif

// hist[0] is the newest class; seen counts windows up to VOTE_MAX. The caller
// owns both as static state (fully partitioned registers).
static void temporal_vote_push(vote_cls_t hist[VOTE_MAX], vote_cnt_t& seen, int cls, int k_in,
                               int& vote_class, int& vote_count) {
#pragma HLS INLINE
    const vote_cnt_t k = k_in < 1 ? 1 : (k_in > VOTE_MAX ? VOTE_MAX : k_in);
SHIFT:
    for (int i = VOTE_MAX - 1; i > 0; --i) {
#pragma HLS UNROLL
        hist[i] = hist[i - 1];
    }
    hist[0] = (vote_cls_t)(cls & 3);
    seen = seen < VOTE_MAX ? (vote_cnt_t)(seen + 1) : seen;
    const vote_cnt_t m = seen < k ? seen : k;
    bool valid[VOTE_MAX];
#pragma HLS ARRAY_PARTITION variable=valid complete
    for (int i = 0; i < VOTE_MAX; ++i) {
#pragma HLS UNROLL
        valid[i] = i < (int)m;
    }
    // Four independent 15-input popcounts.
    vote_cnt_t counts[4];
#pragma HLS ARRAY_PARTITION variable=counts complete
COUNT:
    for (int c = 0; c < 4; ++c) {
#pragma HLS UNROLL
        vote_cnt_t n = 0;
        for (int i = 0; i < VOTE_MAX; ++i) {
#pragma HLS UNROLL
            n += (valid[i] && hist[i] == (vote_cls_t)c) ? 1 : 0;
        }
        counts[c] = n;
    }
    vote_cnt_t best = counts[0];
    for (int c = 1; c < 4; ++c) {
#pragma HLS UNROLL
        best = counts[c] > best ? counts[c] : best;
    }
    bool is_best[4];
#pragma HLS ARRAY_PARTITION variable=is_best complete
    for (int c = 0; c < 4; ++c) {
#pragma HLS UNROLL
        is_best[c] = counts[c] == best;
    }
    // Newest valid entry whose class has the most votes. Scanning oldest -> newest
    // makes the newest match the last one written: a plain select chain. (An
    // early-exit flag scanning newest -> oldest made HLS emit control flow, a
    // 26-state FSM and ~16 k LUT in the top level after Vivado synthesis.)
    vote_cls_t winner = hist[0];
PICK:
    for (int i = VOTE_MAX - 1; i >= 0; --i) {
#pragma HLS UNROLL
        winner = (valid[i] && is_best[hist[i]]) ? hist[i] : winner;
    }
    vote_class = (int)winner;
    vote_count = (int)best;
}

#endif  // TEMPORAL_VOTE_H
