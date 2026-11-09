// Exhaustive unit test of argmax_first_max (the synthesizable class decision).
// Compares against an independent reference (first index of the maximum, the
// numpy.argmax rule) over:
//   * every 4-tuple with values in [-3, 3]  (2401 cases, all tie patterns:
//     2-, 3- and 4-way ties at every position), and
//   * every 4-tuple drawn from the INT10 extremes {-511, -510, 0, 510, 511}.
#include <cstdint>
#include <iostream>
#include "argmax_first_max.h"

static int ref_first_max(const long v[4]) {
    long m = v[0];
    for (int c = 1; c < 4; ++c) if (v[c] > m) m = v[c];
    for (int c = 0; c < 4; ++c) if (v[c] == m) return c;
    return -1;
}

int main() {
    long cases = 0, fails = 0, ties = 0;
    long tie_by_winner[4] = {0, 0, 0, 0};
    auto check = [&](const long v[4]) {
        const int got = argmax_first_max<long, 4>(v);
        const int ref = ref_first_max(v);
        int nmax = 0;
        for (int c = 0; c < 4; ++c) nmax += (v[c] == v[ref]);
        if (nmax > 1) { ++ties; ++tie_by_winner[ref]; }
        ++cases;
        if (got != ref) {
            ++fails;
            std::cout << "FAIL [" << v[0] << "," << v[1] << "," << v[2] << "," << v[3]
                      << "] got=" << got << " ref=" << ref << "\n";
        }
    };
    for (long a = -3; a <= 3; ++a) for (long b = -3; b <= 3; ++b)
    for (long c = -3; c <= 3; ++c) for (long d = -3; d <= 3; ++d) {
        const long v[4] = {a, b, c, d};
        check(v);
    }
    const long ext[5] = {-511, -510, 0, 510, 511};
    for (int a = 0; a < 5; ++a) for (int b = 0; b < 5; ++b)
    for (int c = 0; c < 5; ++c) for (int d = 0; d < 5; ++d) {
        const long v[4] = {ext[a], ext[b], ext[c], ext[d]};
        check(v);
    }
    std::cout << "cases=" << cases << " ties=" << ties << " tie_winner_by_class=["
              << tie_by_winner[0] << "," << tie_by_winner[1] << "," << tie_by_winner[2] << ","
              << tie_by_winner[3] << "] fails=" << fails << "\n";
    std::cout << (fails == 0 ? "ARGMAX UNIT TEST PASS" : "ARGMAX UNIT TEST FAIL") << "\n";
    return fails == 0 ? 0 : 1;
}
