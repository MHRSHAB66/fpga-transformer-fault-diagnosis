# Baseline synthesis of the AXI-Stream top for an explicitly selected test part.
# This is a feasibility probe, not a board-specific implementation or export.
if {![info exists ::env(FPGA_PART)] || $::env(FPGA_PART) eq ""} {
    error "Set FPGA_PART to the exact test part before C-synthesis"
}
set here [file dirname [file normalize [info script]]]

open_project fault_int10_top_csynth
set_top transformer_top
add_files [file join $here transformer_top.cpp]
open_solution sol1 -flow_target vivado
set_part $::env(FPGA_PART)
create_clock -period 10 -name default
csynth_design
exit
