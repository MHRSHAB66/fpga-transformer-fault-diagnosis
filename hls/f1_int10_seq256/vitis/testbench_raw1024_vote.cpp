// Functional test for the RAW-INPUT + TEMPORAL VOTE top.
//
// Runs every vector of <golden-directory> in order (one continuous stream) with
// vote_k = <k> and checks, per vector, the 4 streamed logits (TLAST on the 4th)
// and the pred_class register against golden_expected.txt. It prints
// "VOTE <n> <vote_class> <vote_count>" lines; tools/check_vote_log.py compares them
// with host/temporal_vote.py over the expected classes.
// Usage: testbench_raw1024_vote <golden-directory> <k>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

typedef ap_axis<32, 0, 0, 0> axis_t;
void transformer_top(hls::stream<axis_t>& in, hls::stream<axis_t>& out, int& pred_class,
                     int vote_k, int& vote_class, int& vote_count);

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: testbench_raw1024_vote <golden-directory> <k>\n";
        return 2;
    }
    const std::string dir(argv[1]);
    const int k = std::atoi(argv[2]);
    std::ifstream inputs(dir + "/golden_in.txt");
    std::ifstream expected(dir + "/golden_expected.txt");
    int count = 0, seq = 0, channels = 0;
    if (!inputs || !expected || !(inputs >> count >> seq >> channels) || seq != 1024 || channels != 8) {
        std::cerr << "cannot read golden set\n";
        return 2;
    }
    std::string header;
    std::getline(expected, header);
    int matched = 0;
    for (int n = 0; n < count; ++n) {
        hls::stream<axis_t> in, out;
        for (int i = 0; i < seq * channels; ++i) {
            long long value = 0;
            inputs >> value;
            axis_t word;
            word.data = static_cast<int32_t>(value);
            word.keep = -1;
            word.strb = -1;
            word.last = (i == seq * channels - 1);
            in.write(word);
        }
        int index = 0, true_class = 0, expected_class = 0;
        long long el[4];
        expected >> index >> true_class >> expected_class >> el[0] >> el[1] >> el[2] >> el[3];
        int pred_class = -1, vote_class = -1, vote_count = -1;
        transformer_top(in, out, pred_class, k, vote_class, vote_count);
        bool exact = true;
        for (int c = 0; c < 4; ++c) {
            axis_t word = out.read();
            exact &= static_cast<int32_t>(word.data) == el[c] && static_cast<bool>(word.last) == (c == 3);
        }
        exact &= out.empty() && pred_class == expected_class;
        matched += exact;
        std::cout << "VOTE " << n << " " << vote_class << " " << vote_count
                  << (exact ? "" : " WINDOW_DIFF") << "\n";
    }
    std::cout << "vote top exact vectors (logits + class register): " << matched << "/" << count << "\n";
    return matched == count ? 0 : 1;
}
