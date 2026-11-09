// Synthesizable class decision for the SEQ=256 INT10 core.
//
// Tie policy (fixed, matches the approved reference): scan class ids in
// ascending order and replace the running best only on a STRICTLY greater
// logit, so among equal maxima the LOWEST class id wins. This is exactly
// numpy.argmax (used by the Python INT10 reference, the PYNQ driver and the
// recorded board runs) and the `>` scan in testbench_ingress1024.cpp.
// Class ids: 0 normal, 1 imbalance, 2 misalignment, 3 bearing_fault.
#ifndef ARGMAX_FIRST_MAX_H
#define ARGMAX_FIRST_MAX_H

template <typename T, int N>
static inline int argmax_first_max(const T logits[N]) {
#pragma HLS INLINE
    int best = 0;
    T best_val = logits[0];
    for (int c = 1; c < N; ++c) {
#pragma HLS UNROLL
        if (logits[c] > best_val) {
            best_val = logits[c];
            best = c;
        }
    }
    return best;
}

#endif  // ARGMAX_FIRST_MAX_H
