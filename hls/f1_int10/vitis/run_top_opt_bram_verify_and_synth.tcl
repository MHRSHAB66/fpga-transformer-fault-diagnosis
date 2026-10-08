# Verify (CSim) + C-synthesize the BRAM-ORIENTED optimized top
# (transformer_top_opt_bram.cpp) with Vitis HLS 2020.2.
#
# Goal of this variant: keep the context speedup but remove the weight-ROM
# replication that made transformer_top_opt.cpp (v1) overflow Z7020 BRAM.
#
# Run from a scratch dir (project is created in the CWD; keep it out of git):
#     vitis_hls -f <path>/run_top_opt_bram_verify_and_synth.tcl
#
# Reproducibility: -reset on project AND solution (no stale results reused);
# CSim BEFORE CSynth; CSim compiled with -DHLS_USE_AP_INT so the real HLS widths
# (ap_int<10>/<32>/<20>) from ap_types.h are exercised. Target/clock overridable
# via env FPGA_PART / CLOCK_NS, defaulting to xc7z020clg400-1 @ 10 ns.
set here [file dirname [file normalize [info script]]]
set golden [file normalize [file join $here .. golden]]

set part "xc7z020clg400-1"
if {[info exists ::env(FPGA_PART)] && $::env(FPGA_PART) ne ""} { set part $::env(FPGA_PART) }
set clk 10
if {[info exists ::env(CLOCK_NS)] && $::env(CLOCK_NS) ne ""} { set clk $::env(CLOCK_NS) }

puts "== BRAM variant: top=transformer_top src=transformer_top_opt_bram.cpp part=$part clk=${clk}ns =="

open_project -reset opt_ctx_bram_z7020
set_top transformer_top
add_files [file join $here transformer_top_opt_bram.cpp] -cflags "-DHLS_USE_AP_INT -I$here"
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
