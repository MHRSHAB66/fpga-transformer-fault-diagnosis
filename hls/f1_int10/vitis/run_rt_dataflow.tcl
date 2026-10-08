# Verify (CSim) + C-synthesize the BRAM-oriented v3 top (transformer_top_rt_dataflow.cpp)
# on the PROVISIONAL Zynq-7010 target xc7z020clg400-1, with Vitis HLS 2020.2.
#
# Same model / source / numeric formats / golden vectors / AXI interface /
# sequence length / 10 ns clock as the completed Zynq-7020 v3 measurement. Only
# the target part differs. xc7z020clg400-1 is a PROVISIONAL experimental target,
# NOT a selected board.
#
# Run from a fresh scratch dir (project created in CWD; keep out of git):
#     vitis_hls -f <path>/run_top_opt_bram_z7010_verify_and_synth.tcl
#
# Reproducibility: -reset on project AND solution (no stale results reused);
# CSim BEFORE CSynth; CSim compiled with -DHLS_USE_AP_INT so real HLS widths
# (ap_int<10>/<32>/<20>) from ap_types.h are exercised. Part/clock overridable via
# env FPGA_PART / CLOCK_NS but DEFAULT to xc7z020clg400-1 @ 10 ns for this run.
set here [file dirname [file normalize [info script]]]
set golden [file normalize [file join $here .. golden]]

set part "xc7z020clg400-1"
if {[info exists ::env(FPGA_PART)] && $::env(FPGA_PART) ne ""} { set part $::env(FPGA_PART) }
set clk 10
if {[info exists ::env(CLOCK_NS)] && $::env(CLOCK_NS) ne ""} { set clk $::env(CLOCK_NS) }

puts "== Z7020 rt_dataflow run: top=transformer_top src=transformer_top_rt_dataflow.cpp part=$part clk=${clk}ns =="

open_project -reset rt_dataflow_z7020
set_top transformer_top
add_files [file join $here transformer_top_rt_dataflow.cpp] -cflags "-DHLS_USE_AP_INT -I$here"
add_files -tb [file join $here testbench_top.cpp]
add_files -tb $golden
open_solution -reset sol1 -flow_target vivado
set_part $part
create_clock -period $clk -name default

puts "== CSIM (ap_int widths) start =="
csim_design -O -argv $golden
puts "== CSIM done; CSYNTH start =="
csynth_design
puts "== CSYNTH done =="
exit
