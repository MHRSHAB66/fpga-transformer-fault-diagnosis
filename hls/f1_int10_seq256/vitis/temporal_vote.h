// Temporal majority vote over the last k window classes, inside the FPGA (2026-11-10).
//
// Bit-exact reference: host/temporal_vote.py. The core keeps the classes of the
// last VOTE_MAX windows (registers, initialised empty at bitstream load, never
// reset afterwards: a continuous stream). After each window it votes over the
// newest m = min(seen, k) entries, k = vote_k clamped to 1..VOTE_MAX; most votes
// wins, ties go to the class seen most recently. Outputs: winning class and its
// vote count. A few dozen LUTs; no DSP, no BRAM.
#ifndef TEMPORAL_VOTE_H
#define TEMPORAL_VOTE_H

#define VOTE_MAX 15

// hist[0] is the newest class; seen counts windows up to VOTE_MAX. The caller
// owns both as static state (fully partitioned registers).
static void temporal_vote_push(int hist[VOTE_MAX], int& seen, int cls, int k_in,
                               int& vote_class, int& vote_count) {
#pragma HLS INLINE
    int k = k_in < 1 ? 1 : (k_in > VOTE_MAX ? VOTE_MAX : k_in);
SHIFT:
    for (int i = VOTE_MAX - 1; i > 0; --i) {
#pragma HLS UNROLL
        hist[i] = hist[i - 1];
    }
    hist[0] = cls;
    if (seen < VOTE_MAX) seen = seen + 1;
    const int m = seen < k ? seen : k;
    int counts[4] = {0, 0, 0, 0};
#pragma HLS ARRAY_PARTITION variable=counts complete
COUNT:
    for (int i = 0; i < VOTE_MAX; ++i) {
#pragma HLS UNROLL
        if (i < m) counts[hist[i] & 3] += 1;
    }
    int best = counts[0];
    for (int c = 1; c < 4; ++c) {
#pragma HLS UNROLL
        if (counts[c] > best) best = counts[c];
    }
    int winner = hist[0] & 3;
    bool found = false;
PICK:
    for (int i = 0; i < VOTE_MAX; ++i) {
#pragma HLS UNROLL
        const int c = hist[i] & 3;
        if (!found && i < m && counts[c] == best) {
            winner = c;
            found = true;
        }
    }
    vote_class = winner;
    vote_count = best;
}

#endif  // TEMPORAL_VOTE_H
