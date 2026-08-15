// Loader for the exported model constants (hls/data/). Testbench-side only —
// on a real board these become BRAM initialisers or AXI-streamed weights.
#include "transformer.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

static std::vector<int64_t> read_ints(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    size_t n; in >> n;
    std::vector<int64_t> v(n);
    for (size_t i = 0; i < n; ++i) in >> v[i];
    return v;
}

static std::vector<double> read_floats(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    size_t n; in >> n;
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i) in >> v[i];
    return v;
}

static std::string L(int l) { return "l" + std::to_string(l); }

Params load_params(const std::string& dir) {
    Params p;
    auto D_ = dir + "/";

    // linear layers: weights, folded biases, requant multipliers
    std::vector<std::string> linears = {"proj", "cls1", "cls4"};
    for (int l = 0; l < LAYERS; ++l) {
        linears.push_back(L(l) + ".wq");
        linears.push_back(L(l) + ".wk");
        linears.push_back(L(l) + ".wv");
        linears.push_back(L(l) + ".out");
        linears.push_back(L(l) + ".ff0");
        linears.push_back(L(l) + ".ff3");
    }
    for (auto& name : linears) {
        p.W[name] = read_ints(D_ + "W_" + name + ".txt");
        p.b[name] = read_ints(D_ + "b_" + name + ".txt");
        p.M[name] = read_floats(D_ + "M_" + name + ".txt")[0];
    }

    // attention + pool requant multipliers
    for (int l = 0; l < LAYERS; ++l) {
        p.M[L(l) + ".scores"] = read_floats(D_ + "M_" + L(l) + ".scores.txt")[0];
        p.M[L(l) + ".ctx"]    = read_floats(D_ + "M_" + L(l) + ".ctx.txt")[0];
    }
    p.M["pool"] = read_floats(D_ + "M_pool.txt")[0];

    // residual add ratios
    for (int l = 0; l < LAYERS; ++l) {
        p.add[L(l) + ".res1"] = read_floats(D_ + "add_" + L(l) + ".res1.txt");
        p.add[L(l) + ".res2"] = read_floats(D_ + "add_" + L(l) + ".res2.txt");
    }

    // positional encoding
    p.pe = read_ints(D_ + "pe_codes.txt");

    // LayerNorm affine + eps
    std::vector<std::string> lns = {"cls_ln"};
    for (int l = 0; l < LAYERS; ++l) { lns.push_back(L(l) + ".ln1"); lns.push_back(L(l) + ".ln2"); }
    for (auto& name : lns) {
        p.ln_gm[name]   = read_floats(D_ + "ln_gm_" + name + ".txt");
        p.ln_bt[name]   = read_floats(D_ + "ln_bt_" + name + ".txt");
        p.ln_epsq[name] = read_ints(D_ + "ln_epsq_" + name + ".txt")[0];
    }

    // LUTs
    for (int l = 0; l < LAYERS; ++l) {
        p.gelu[l]      = read_ints(D_ + "gelu_" + L(l) + ".txt");
        p.exp_lut[l]   = read_ints(D_ + "exp_" + L(l) + ".txt");
        p.exp_clamp[l] = (int)read_ints(D_ + "exp_clamp_" + L(l) + ".txt")[0];
    }
    p.gelu_cls = read_ints(D_ + "gelu_cls.txt");
    p.rsqrt    = read_ints(D_ + "rsqrt.txt");

    return p;
}
