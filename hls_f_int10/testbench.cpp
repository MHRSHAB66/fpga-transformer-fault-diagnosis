// ---------------------------------------------------------------------------
// Testbench: run the integer core over the exported golden vectors and check it
// reproduces the Python reference bit-for-bit (class + every logit).
//
//   g++ -O2 -std=c++17 transformer.cpp params_io.cpp testbench.cpp -o tb
//   ./tb            # uses ./data and ./golden by default
// ---------------------------------------------------------------------------
#include "transformer.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static const char* CLASS_NAMES[NUM_CLASSES] = {"normal", "imbalance", "misalignment", "bearing_fault"};

int main(int argc, char** argv) {
    std::string data_dir   = (argc > 1) ? argv[1] : "data";
    std::string golden_dir = (argc > 2) ? argv[2] : "golden";

    std::cout << "Loading params from " << data_dir << " ...\n";
    Params p = load_params(data_dir);

    // --- read golden inputs -------------------------------------------------
    std::ifstream fin(golden_dir + "/golden_in.txt");
    if (!fin) { std::cerr << "cannot open golden_in.txt\n"; return 2; }
    int N, seq, ch; fin >> N >> seq >> ch;
    if (seq != SEQ || ch != CH) { std::cerr << "golden geometry mismatch\n"; return 2; }
    std::vector<std::vector<int64_t>> inputs(N, std::vector<int64_t>(SEQ * CH));
    for (int n = 0; n < N; ++n)
        for (int i = 0; i < SEQ * CH; ++i) fin >> inputs[n][i];

    // --- read expected outputs ---------------------------------------------
    std::ifstream fexp(golden_dir + "/golden_expected.txt");
    if (!fexp) { std::cerr << "cannot open golden_expected.txt\n"; return 2; }
    std::string header; std::getline(fexp, header);
    std::vector<int> exp_true(N), exp_pred(N);
    std::vector<std::vector<int64_t>> exp_logits(N, std::vector<int64_t>(NUM_CLASSES));
    for (int n = 0; n < N; ++n) {
        int idx; fexp >> idx >> exp_true[n] >> exp_pred[n];
        for (int c = 0; c < NUM_CLASSES; ++c) fexp >> exp_logits[n][c];
    }

    // --- run core -----------------------------------------------------------
    std::cout << "\n  #  true             predicted         class  logits_match\n";
    std::cout << "  ---------------------------------------------------------------\n";
    int class_ok = 0, logit_ok = 0;
    for (int n = 0; n < N; ++n) {
        int64_t logits[NUM_CLASSES];
        transformer_infer(inputs[n].data(), p, logits);
        int pred = 0;
        for (int c = 1; c < NUM_CLASSES; ++c) if (logits[c] > logits[pred]) pred = c;

        bool cmatch = (pred == exp_pred[n]);
        bool lmatch = true;
        for (int c = 0; c < NUM_CLASSES; ++c) if (logits[c] != exp_logits[n][c]) lmatch = false;
        class_ok += cmatch; logit_ok += lmatch;

        std::cout << "  " << (n < 10 ? " " : "") << n << "  "
                  << CLASS_NAMES[exp_true[n]] << std::string(16 - std::string(CLASS_NAMES[exp_true[n]]).size(), ' ')
                  << " " << CLASS_NAMES[pred] << std::string(16 - std::string(CLASS_NAMES[pred]).size(), ' ')
                  << "  " << (cmatch ? "ok " : "ERR")
                  << "  " << (lmatch ? "exact" : "DIFF") << "\n";
    }

    std::cout << "\n  class predictions : " << class_ok << "/" << N << "\n";
    std::cout << "  bit-exact logits  : " << logit_ok << "/" << N << "\n";
    bool pass = (class_ok == N && logit_ok == N);
    std::cout << "  RESULT: " << (pass ? "PASS" : "FAIL") << "\n";
    return pass ? 0 : 1;
}
