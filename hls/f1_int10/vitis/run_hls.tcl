# Vitis HLS project script for the F_1layer @ int10 core.
#
# Target device is intentionally not hard-coded. Set FPGA_PART to the exact
# Vitis part only after the final board/device is selected.
#
# Usage (from hls/f1_int10/vitis/, with vitis_hls on PATH):
#   vitis_hls -f run_hls.tcl              # runs C-simulation of the golden core
#
# Stages are gated by variables at the top so you can enable csynth/cosim/export
# once the synthesizable top (transformer_top.cpp) is ready. C-SIM WORKS NOW: it
# builds the portable core (transformer.cpp + params_io.cpp) with the golden
# testbench and must print "RESULT: PASS" (16/16), matching the NumPy verify.

# ---- config ---------------------------------------------------------------
set HLSDIR [file normalize [pwd]/..]            ;# repo/hls/f1_int10
if {![info exists ::env(FPGA_PART)] || $::env(FPGA_PART) eq ""} {
    error "FPGA_PART is not set; select the final FPGA device before running Vitis HLS"
}
set PART   $::env(FPGA_PART)
set PERIOD 10                                   ;# ns -> 100 MHz starting target

set DO_CSIM   1
set DO_CSYNTH 0    ;# enable after transformer_top.cpp is completed
set DO_COSIM  0
set DO_EXPORT 0

# ---------------------------------------------------------------------------
open_project fault_int10
open_solution "sol1" -flow_target vivado
set_part $PART
create_clock -period $PERIOD -name default

# --- C-SIM: validate the golden core inside the Vitis flow -----------------
# Portable core + text-file param loader + golden testbench. Not synthesizable
# (uses std::map/file IO) but perfect for functional C-simulation.
set_top transformer_infer
add_files            $HLSDIR/transformer.cpp
add_files            $HLSDIR/params_io.cpp
add_files -tb        $HLSDIR/testbench.cpp
add_files -tb        $HLSDIR/data
add_files -tb        $HLSDIR/golden

if {$DO_CSIM} {
    # testbench reads <argv1>=data <argv2>=golden
    csim_design -argv "$HLSDIR/data $HLSDIR/golden"
}

# --- C-SYNTH: the synthesizable top (see SYNTHESIS_GUIDE.md) ----------------
# Swap the design files to the synthesizable core once ready:
#   set_top transformer_top
#   add_files $HLSDIR/vitis/transformer_top.cpp   ;# includes weights_int10.h
# (weights_int10.h replaces the file loader; ap_types.h narrows the datapath.)
if {$DO_CSYNTH} {
    csynth_design
}
if {$DO_COSIM} {
    cosim_design -argv "$HLSDIR/data $HLSDIR/golden"
}
if {$DO_EXPORT} {
    export_design -format ip_catalog -rtl verilog
}

exit
