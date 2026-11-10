// g++ check of temporal_vote.h: reads "k class" pairs (one per line) from stdin,
// runs the continuous voter and prints "vote_class vote_count" per line.
// tests/test_temporal_vote.py compares the output with host/temporal_vote.py.
#include <cstdio>

#include "temporal_vote.h"

int main() {
    int hist[VOTE_MAX] = {0};
    int seen = 0;
    int k = 0, cls = 0;
    while (std::scanf("%d %d", &k, &cls) == 2) {
        int vc = -1, cnt = -1;
        temporal_vote_push(hist, seen, cls, k, vc, cnt);
        std::printf("%d %d\n", vc, cnt);
    }
    return 0;
}
