// Minimal desktop g++ stub for hls::stream, ONLY for compiler-based functional
// C-simulation of the AXI-Stream top on a machine without Vitis HLS installed.
//
// This is NOT a synthesis header and is NOT on the synthesizable path. Vitis
// HLS provides its own hls_stream.h during real csim/csynth. This stub lets the
// exact same transformer_top.cpp / testbench_top.cpp compile and run under g++
// so a bit-exact functional comparison (golden vectors) can be made here.
#ifndef GPP_CSIM_HLS_STREAM_H
#define GPP_CSIM_HLS_STREAM_H
#include <deque>
#include <string>
namespace hls {
template <typename T>
class stream {
    std::deque<T> q_;
public:
    stream() = default;
    explicit stream(const char*) {}
    explicit stream(const std::string&) {}
    void write(const T& v) { q_.push_back(v); }
    bool write_nb(const T& v) { q_.push_back(v); return true; }
    T read() { T v = q_.front(); q_.pop_front(); return v; }
    bool read_nb(T& v) { if (q_.empty()) return false; v = q_.front(); q_.pop_front(); return true; }
    bool empty() const { return q_.empty(); }
    size_t size() const { return q_.size(); }
};
}  // namespace hls
#endif
