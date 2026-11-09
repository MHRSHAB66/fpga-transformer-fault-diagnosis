// CPU baseline: the deployed SEQ=256 INT10 top (argmax variant) compiled for a
// general-purpose CPU and timed per window.
//
// The source is the same HLS C++ the FPGA was synthesized from (non-ap_int branch
// of ap_types.h, native integers), so its logits are bit-exact with the board.
// Windows are loaded into memory first; the timer covers only the
// transformer_top call (4:1 ingress aggregation + transformer + argmax), not file
// I/O or filling the input stream.
//
// Usage:  cpu_baseline_bench <raw_int32_file> <n_windows> <repeats>
//   raw file: n_windows x (1024 x 8) little-endian int32, row-major [sample, channel]
// stdout: one line per window of the first repeat: <i> <l0> <l1> <l2> <l3> <pred> <us>
// stderr: SUMMARY n=<inferences> mean_us=.. p50_us=.. p95_us=.. max_us=.. min_us=..
#include <ap_axi_sdata.h>
#include <hls_stream.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

typedef ap_axis<32, 0, 0, 0> axis_t;
void transformer_top(hls::stream<axis_t>& in, hls::stream<axis_t>& out, int& pred_class);

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: cpu_baseline_bench <raw_int32_file> <n_windows> <repeats>\n");
        return 2;
    }
    const long n = std::atol(argv[2]);
    const long repeats = std::atol(argv[3]);
    const int words = 1024 * 8;
    std::vector<int32_t> data((size_t)n * words);
    FILE* f = std::fopen(argv[1], "rb");
    if (!f || std::fread(data.data(), sizeof(int32_t), data.size(), f) != data.size()) {
        std::fprintf(stderr, "cannot read %ld windows from %s\n", n, argv[1]);
        return 3;
    }
    std::fclose(f);

    std::vector<double> us;
    us.reserve((size_t)(n * repeats));
    for (long r = -1; r < repeats; ++r) {          // r = -1: untimed warm-up pass
        for (long i = 0; i < n; ++i) {
            hls::stream<axis_t> in, out;
            const int32_t* w = &data[(size_t)i * words];
            for (int k = 0; k < words; ++k) {
                axis_t a;
                a.data = w[k];
                a.keep = -1; a.strb = -1; a.user = 0; a.id = 0; a.dest = 0;
                a.last = (k == words - 1);
                in.write(a);
            }
            int pred = -1;
            const auto t0 = std::chrono::steady_clock::now();
            transformer_top(in, out, pred);
            const auto t1 = std::chrono::steady_clock::now();
            long long lg[4];
            for (int c = 0; c < 4; ++c) lg[c] = (int32_t)out.read().data;
            const double dt = std::chrono::duration<double, std::micro>(t1 - t0).count();
            if (r >= 0) us.push_back(dt);
            if (r == 0)
                std::printf("%ld %lld %lld %lld %lld %d %.1f\n", i, lg[0], lg[1], lg[2], lg[3],
                            pred, dt);
        }
    }
    std::fflush(stdout);
    std::vector<double> s(us);
    std::sort(s.begin(), s.end());
    double sum = 0;
    for (double v : s) sum += v;
    auto pct = [&](double p) { return s[(size_t)std::min<double>(s.size() - 1, p * (s.size() - 1) + 0.5)]; };
    std::fprintf(stderr, "SUMMARY n=%zu mean_us=%.1f p50_us=%.1f p95_us=%.1f max_us=%.1f min_us=%.1f\n",
                 s.size(), sum / s.size(), pct(0.50), pct(0.95), s.back(), s.front());
    return 0;
}
