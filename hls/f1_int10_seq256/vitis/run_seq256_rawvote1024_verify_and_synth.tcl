# RAW-INPUT + TEMPORAL VOTE variant (2026-11-10): the raw-input core plus a
# majority vote over the last vote_k window classes (temporal_vote.h) on extra
# AXI-Lite registers. Separate project name; the raw-input project is untouched.
# C-simulation runs the raw golden set as one stream with vote_k = 5.
# Usage: vitis_hls -f run_seq256_rawvote1024_verify_and_synth.tcl
set here [file dirname [file normalize [info script]]]
set golden [file normalize [file join $here .. golden_raw1024_argmax]]
set part "xc7z020clg400-1"
set clk 10
set project_name "seq256_rawvote1024"
puts "== SEQ256 DSP-margin + on-FPGA preprocessing + argmax + temporal vote: part=$part clk=${clk}ns =="
open_project -reset $project_name
set_top transformer_top
add_files [file join $here transformer_top_seq256_raw1024_dsp_margin_vote.cpp] \
    -cflags "-DHLS_USE_AP_INT -I$here"
add_files -tb [file join $here testbench_raw1024_vote.cpp]
add_files -tb $golden
open_solution -reset sol1 -flow_target vivado
set_part $part
create_clock -period $clk -name default
csim_design -O -argv "$golden 5"
csynth_design
export_design -format ip_catalog
exit
