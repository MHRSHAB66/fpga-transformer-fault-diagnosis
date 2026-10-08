// Functional test for the AXI-Stream HLS top against exported golden vectors.
#include <ap_axi_sdata.h>
#include <hls_stream.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

typedef ap_axis<32, 0, 0, 0> axis_t;
void transformer_top(hls::stream<axis_t>& in, hls::stream<axis_t>& out);

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: testbench_top <golden-directory>\n";
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
    int matched = 0;
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
        int64_t expected_logits[4];
        if (!(expected >> index >> true_class >> expected_class >> expected_logits[0]
                       >> expected_logits[1] >> expected_logits[2] >> expected_logits[3]) || index != n) {
            std::cerr << "invalid expected output at vector " << n << "\n";
            return 2;
        }
        transformer_top(in, out);
        bool exact = true;
        int32_t logits[4];
        for (int c = 0; c < 4; ++c) {
            if (out.empty()) {
                std::cerr << "missing output at vector " << n << ", logit " << c << "\n";
                return 1;
            }
            axis_t word = out.read();
            logits[c] = static_cast<int32_t>(word.data);
            exact &= logits[c] == expected_logits[c] && static_cast<bool>(word.last) == (c == 3);
        }
        if (!out.empty()) {
            std::cerr << "extra output at vector " << n << "\n";
            return 1;
        }
        int predicted = 0;
        for (int c = 1; c < 4; ++c) if (logits[c] > logits[predicted]) predicted = c;
        exact &= predicted == expected_class;
        matched += exact;
        std::cout << "vector " << n << ": " << (exact ? "exact" : "DIFF")
                  << " expected=[" << expected_logits[0] << "," << expected_logits[1]
                  << "," << expected_logits[2] << "," << expected_logits[3]
                  << "] actual=[" << logits[0] << "," << logits[1] << ","
                  << logits[2] << "," << logits[3] << "]\n";
    }
    std::cout << "AXI top exact vectors: " << matched << "/" << count << "\n";
    return matched == count ? 0 : 1;
}
