# Verify (CSim) + C-synthesize the OPTIMIZED context-accumulation top with
# Vitis HLS 2020.2. Mirrors the frozen baseline flow
# (artifacts/results/hls_z7020_q40_bounded_snapshot/run_verify_and_synth.tcl)
# but set_top uses transformer_top_opt.cpp, NOT the baseline transformer_top.cpp.
#
# Run (project is created in the CURRENT working directory; run from a scratch
# dir so it stays out of git, e.g. repo/tmp/hls_opt_run):
#     vitis_hls -f <path>/run_top_opt_verify_and_synth.tcl
#
# Reproducibility:
#   * open_project -reset / open_solution -reset force a clean project+solution,
#     so no stale CSim/CSynth result is ever reused.
#   * CSim runs BEFORE CSynth. CSim uses -DHLS_USE_AP_INT so the real HLS widths
#     from ap_types.h (ap_int<10>/ap_int<32>/ap_int<20>) are exercised, not the
#     desktop native-width fallback.
#   * Target part / clock are overridable via env FPGA_PART / CLOCK_NS but default
#     to the requested measurement point xc7z020clg400-1 @ 10 ns.
set here [file dirname [file normalize [info script]]]
set golden [file normalize [file join $here .. golden]]

set part "xc7z020clg400-1"
if {[info exists ::env(FPGA_PART)] && $::env(FPGA_PART) ne ""} { set part $::env(FPGA_PART) }
set clk 10
if {[info exists ::env(CLOCK_NS)] && $::env(CLOCK_NS) ne ""} { set clk $::env(CLOCK_NS) }

puts "== OPT run: top=transformer_top src=transformer_top_opt.cpp part=$part clk=${clk}ns =="

open_project -reset opt_ctx_q40_z7020
set_top transformer_top
add_files [file join $here transformer_top_opt.cpp] -cflags "-DHLS_USE_AP_INT -I$here"
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
