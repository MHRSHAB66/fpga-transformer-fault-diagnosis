// Day-2 numeric-contract harness for the DEPLOYED SEQ=256 INT10 HLS top.
//
// The deployed source is #included verbatim, so every command below calls the
// real static primitives (div_round_even, requant, residual, layernorm,
// rsqrt_fxp, gelu_lut, mul_s64_shift_add, df_head<0>, transformer_top) -- no
// copy of the arithmetic exists here. Driven by tests/test_int10_numeric_contract.py.
//
// Build (desktop g++, native wide integers -- the same build as batch_logits):
//   g++ -O2 -std=c++17 -Wno-unknown-pragmas -I hls/f1_int10/gpp_csim/stubs
//       -I hls/f1_int10_seq256/vitis tests/cpp/int10_contract_harness.cpp -o harness
//
// stdin: one command per line, stdout: one result line per command.
//   div N            div_round_even<2^40>(N)
//   req ACC MQ       requant(ACC, MQ)
//   mul A B          mul_s64_shift_add(A, B)
//   rsq V            rsqrt_fxp(V, rsqrt_lut)
//   gelu Q | gelc Q  gelu_lut(Q, gelu_l0 | gelu_cls)
//   res1 A B|res2 A B residual(.., add_l0_res{1,2}_q40)
//   ln1|ln2|lnc x0..x31            layernorm row with the deployed constants
//   head q[8] k[256*8] v[256*8]     df_head<0> -> ctx[8]
//   top w[1024*8]                   transformer_top -> 4 logits
#include "transformer_top_seq256_ingress1024_dsp_margin.cpp"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

template <typename T>
static void read_n(std::istringstream& is, T* dst, long n) {
    for (long i = 0; i < n; ++i) { long long v; is >> v; dst[i] = (T)v; }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::string line;
    static act_t kbuf[SEQ * HD], vbuf[SEQ * HD];
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream is(line);
        std::string cmd; is >> cmd;
        if (cmd == "div") {
            long long n; is >> n;
            std::printf("%lld\n", div_round_even<1LL << SCALE_FRAC_BITS>(n));
        } else if (cmd == "req") {
            long long a, m; is >> a >> m;
            std::printf("%ld\n", requant((long)a, (scale_t)m));
        } else if (cmd == "mul") {
            long long a, m; is >> a >> m;
            std::printf("%lld\n", mul_s64_shift_add((long)a, (scale_t)m));
        } else if (cmd == "rsq") {
            long long v; is >> v;
            std::printf("%ld\n", rsqrt_fxp((long)v, rsqrt_lut));
        } else if (cmd == "gelu" || cmd == "gelc") {
            long long q; is >> q;
            std::printf("%ld\n", gelu_lut((long)q, cmd == "gelu" ? gelu_l0 : gelu_cls));
        } else if (cmd == "res1" || cmd == "res2") {
            long long a, b; is >> a >> b;
            act_t x[1] = {(act_t)a}, y[1] = {(act_t)b}, o[1];
            const scale_t* r = cmd == "res1" ? add_l0_res1_q40 : add_l0_res2_q40;
            residual(x, y, 1, r[0], r[1], o);
            std::printf("%lld\n", (long long)o[0]);
        } else if (cmd == "ln1" || cmd == "ln2" || cmd == "lnc") {
            act_t x[D], o[D];
            read_n(is, x, D);
            if (cmd == "ln1") layernorm(x, 1, ln_gm_l0_ln1_q40, ln_bt_l0_ln1_q40, ln_epsq_l0_ln1, rsqrt_lut, o);
            else if (cmd == "ln2") layernorm(x, 1, ln_gm_l0_ln2_q40, ln_bt_l0_ln2_q40, ln_epsq_l0_ln2, rsqrt_lut, o);
            else layernorm(x, 1, ln_gm_cls_ln_q40, ln_bt_cls_ln_q40, ln_epsq_cls_ln, rsqrt_lut, o);
            for (int j = 0; j < D; ++j) std::printf(j ? " %lld" : "%lld", (long long)o[j]);
            std::printf("\n");
        } else if (cmd == "head") {
            act_t q[HD], c[HD];
            read_n(is, q, HD); read_n(is, kbuf, SEQ * HD); read_n(is, vbuf, SEQ * HD);
            df_head<0>(q, kbuf, vbuf, c);
            for (int d = 0; d < HD; ++d) std::printf(d ? " %lld" : "%lld", (long long)c[d]);
            std::printf("\n");
        } else if (cmd == "top") {
            hls::stream<axis_t> in, out;
            for (int i = 0; i < INPUT_SEQ * CH; ++i) {
                long long v; is >> v;
                axis_t w; w.data = (int)v; w.keep = -1; w.strb = -1;
                w.user = 0; w.id = 0; w.dest = 0; w.last = (i == INPUT_SEQ * CH - 1);
                in.write(w);
            }
            transformer_top(in, out);
            for (int c = 0; c < NUM_CLASSES; ++c) {
                axis_t t = out.read();
                std::printf(c ? " %d" : "%d", (int)t.data);
            }
            std::printf("\n");
        } else {
            std::printf("ERR %s\n", cmd.c_str());
        }
    }
    return 0;
}
