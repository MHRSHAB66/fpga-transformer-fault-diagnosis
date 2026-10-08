// Minimal desktop g++ stub for ap_axis, ONLY for compiler-based functional
// C-simulation on a machine without Vitis HLS installed. NOT a synthesis header
// and NOT on the synthesizable path. Vitis HLS provides the real ap_axi_sdata.h
// during csim/csynth. Fields mirror the members used by the top/testbench
// (data/keep/strb/last) with wide native integers so the int10 arithmetic is
// preserved exactly under g++ (which uses the non-ap_int branch of ap_types.h).
#ifndef GPP_CSIM_AP_AXI_SDATA_H
#define GPP_CSIM_AP_AXI_SDATA_H
#include <cstdint>
template <int D, int U, int TI, int TD>
struct ap_axis {
    int64_t data;
    int64_t keep;
    int64_t strb;
    int64_t user;
    int64_t last;
    int64_t id;
    int64_t dest;
};
#endif
