// Functional test for the ARGMAX variant of the SEQ=256 ingress-1024 AXI top.
//
// For every vector in <golden-directory>:
//   * the 4 streamed logits must equal golden_expected.txt exactly, TLAST on
//     the 4th word and no extra words (output contract v1 unchanged);
//   * the hardware class register `pred_class` must equal the expected class
//     column AND the lowest-index argmax of the expected logits (tie policy).
// Works with the existing golden_ingress1024/ set and with generated sets that
// add tie windows (see tools/make_argmax_golden.py).
#include <ap_axi_sdata.h>
#include <hls_stream.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

typedef ap_axis<32, 0, 0, 0> axis_t;
void transformer_top(hls::stream<axis_t>& in, hls::stream<axis_t>& out, int& pred_class);

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: testbench_ingress1024_argmax <golden-directory>\n";
        return 2;
    }
    const std::string dir(argv[1]);
    std::ifstream inputs(dir + "/golden_in.txt");
    std::ifstream expected(dir + "/golden_expected.txt");
    if (!inputs || !expected) {
        std::cerr << "cannot open golden input/expected files\n";
        return 2;
    }
    int count = 0, seq = 0, channels = 0;
    if (!(inputs >> count >> seq >> channels) || count <= 0 || seq != 1024 || channels != 8) {
        std::cerr << "invalid golden input geometry\n";
        return 2;
    }
    std::string header;
    std::getline(expected, header);
    int matched = 0, ties = 0, per_class[4] = {0, 0, 0, 0};
    for (int n = 0; n < count; ++n) {
        hls::stream<axis_t> in, out;
        for (int i = 0; i < seq * channels; ++i) {
            int64_t value = 0;
            if (!(inputs >> value) || value < std::numeric_limits<int32_t>::min() ||
                value > std::numeric_limits<int32_t>::max()) {
                std::cerr << "invalid input at vector " << n << ", sample " << i << "\n";
                return 2;
            }
            axis_t word;
            word.data = static_cast<int32_t>(value);
            word.keep = -1;
            word.strb = -1;
            word.last = (i == seq * channels - 1);
            in.write(word);
        }
        int index = 0, true_class = 0, expected_class = 0;
        int64_t el[4];
        if (!(expected >> index >> true_class >> expected_class >> el[0] >> el[1] >> el[2] >> el[3]) ||
            index != n) {
            std::cerr << "invalid expected output at vector " << n << "\n";
            return 2;
        }
        // reference decision from the expected logits: lowest index among maxima
        int ref = 0;
        for (int c = 1; c < 4; ++c) if (el[c] > el[ref]) ref = c;
        int nmax = 0;
        for (int c = 0; c < 4; ++c) nmax += (el[c] == el[ref]);
        const bool tie = nmax > 1;
        ties += tie;

        int pred_class = -1;
        transformer_top(in, out, pred_class);
        bool exact = true;
        int32_t lg[4];
        for (int c = 0; c < 4; ++c) {
            if (out.empty()) {
                std::cerr << "missing output at vector " << n << ", logit " << c << "\n";
                return 1;
            }
            axis_t word = out.read();
            lg[c] = static_cast<int32_t>(word.data);
            exact &= lg[c] == el[c] && static_cast<bool>(word.last) == (c == 3);
        }
        if (!out.empty()) {
            std::cerr << "extra output at vector " << n << "\n";
            return 1;
        }
        exact &= pred_class == expected_class && pred_class == ref;
        if (exact && pred_class >= 0 && pred_class < 4) per_class[pred_class]++;
        matched += exact;
        std::cout << "vector " << n << ": " << (exact ? "exact" : "DIFF")
                  << (tie ? " TIE" : "") << " hw_class=" << pred_class
                  << " expected_class=" << expected_class
                  << " logits=[" << lg[0] << "," << lg[1] << "," << lg[2] << "," << lg[3] << "]\n";
    }
    std::cout << "per-class hw decisions: normal=" << per_class[0] << " imbalance=" << per_class[1]
              << " misalignment=" << per_class[2] << " bearing_fault=" << per_class[3] << "\n";
    std::cout << "tie vectors: " << ties << "\n";
    std::cout << "AXI top exact vectors (logits + class register): " << matched << "/" << count << "\n";
    return matched == count ? 0 : 1;
}
