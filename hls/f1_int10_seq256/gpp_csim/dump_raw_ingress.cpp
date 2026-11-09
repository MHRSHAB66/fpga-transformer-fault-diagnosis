// g++ C-model check of the on-FPGA preprocessing (onchip_preprocess.h).
//
// Reads raw windows (little-endian int32, 1024 x 8 row-major) from a file and
// prints, per window, the 256 x 8 INT10 tokens raw_ingress hands to the
// transformer. tools/verify_onchip_preprocess.py compares them with the Python
// reference (src/quantize/onchip_preprocess.py + aggregate_round_div).
//
// Usage: dump_raw_ingress <raw_int32_file> <n_windows>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "onchip_preprocess.h"

typedef ap_axis<32, 0, 0, 0> axis_t;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: dump_raw_ingress <raw_int32_file> <n_windows>\n");
        return 2;
    }
    const long n = std::atol(argv[2]);
    const int words = 1024 * 8;
    std::vector<int32_t> buf(words);
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return 3;
    for (long w = 0; w < n; ++w) {
        if (std::fread(buf.data(), sizeof(int32_t), words, f) != (size_t)words) return 3;
        hls::stream<axis_t> in;
        for (int i = 0; i < words; ++i) {
            axis_t a;
            a.data = buf[i];
            a.keep = -1; a.strb = -1; a.user = 0; a.id = 0; a.dest = 0;
            a.last = (i == words - 1);
            in.write(a);
        }
        long long tokens[256 * 8];
        raw_ingress<axis_t, long long, 1024, 256, 8>(in, tokens);
        for (int i = 0; i < 256 * 8; ++i) std::printf(i ? " %lld" : "%lld", tokens[i]);
        std::printf("\n");
    }
    std::fclose(f);
    return 0;
}
