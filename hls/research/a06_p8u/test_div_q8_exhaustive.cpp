// A06 exactness check: div_q8 == the P8-unsigned reference quotient for EVERY reachable
// (num, denom): num over all distinct exp_l0 values, denom over the whole
// interval [32768, 256*32768] (a superset of the reachable row sums).
// Build (native and ap_int):
//   g++ -O2 -std=c++17 -I hls/f1_int10_seq256/vitis -I hls/research/a04_softmax_div \
//       hls/research/a04_softmax_div/test_div_q9_exhaustive.cpp -o tmp/a04_exh.exe
//   g++ -O2 -std=c++14 -DHLS_USE_AP_INT -I G:/Xilinx/Vitis_HLS/2020.2/include ... (ap_int build)
#include <cstdio>
#include <cstdlib>
#include <set>

#include "ap_types.h"
#include "weights_int10.h"
#include "softmax_div_q8.h"

int main(int argc, char** argv) {
    // ap_int build: sample every STRIDE-th denominator (the ap_int types are slow);
    // native build: every denominator.
    const long long stride = argc > 1 ? std::atoll(argv[1]) : 1;
    std::set<long> nums(exp_l0, exp_l0 + exp_clamp_l0 + 1);
    const long long dmin = 32768, dmax = 256LL * 32768;
    long long checked = 0, mismatches = 0, max_q = 0, min_slack = 1LL << 62;
    for (long num : nums) {
        if (num < 0) { std::printf("negative exp entry\n"); return 1; }
        for (long long denom = dmin; denom <= dmax; denom += stride) {
            if (num > denom) continue;   // unreachable: num is one term of denom
            const long long numer = (long long)num * 255 + denom / 2;           // canonical numerator
            const long long canon = numer / denom;                               // canonical quotient
            if (numer >= (1LL << 24) || denom >= (1LL << 24) || numer >= 256 * denom) {
                std::printf("range violation num=%ld denom=%lld\n", num, denom);
                return 1;
            }
            const long long got = (long long)div_q8((q8_numer_t)numer, (q8_denom_t)denom);
            if (got != canon) {
                if (mismatches < 5) std::printf("MISMATCH num=%ld denom=%lld canon=%lld got=%lld\n", num, denom, canon, got);
                ++mismatches;
            }
            if (canon > max_q) max_q = canon;
            const long long slack = 256 * denom - numer;
            if (slack < min_slack) min_slack = slack;
            ++checked;
        }
    }
    std::printf("distinct_num=%zu checked=%lld mismatches=%lld max_quotient=%lld min(256*denom-numer)=%lld stride=%lld\n",
                nums.size(), checked, mismatches, max_q, min_slack, stride);
    return mismatches ? 2 : 0;
}
