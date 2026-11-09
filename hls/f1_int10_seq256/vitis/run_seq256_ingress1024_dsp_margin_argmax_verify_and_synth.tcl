# ARGMAX variant (2026-10-22): same core as the deployed DSP-margin top plus a
# synthesizable class decision exposed on AXI-Lite register `pred_class`.
# Not yet run: requires Vitis HLS (laptop). Separate project name, so the
# deployed seq256_ingress1024_dsp_margin project/IP is never overwritten.
# Usage: vitis_hls -f run_seq256_ingress1024_dsp_margin_argmax_verify_and_synth.tcl
set here [file dirname [file normalize [info script]]]
set golden [file normalize [file join $here .. golden_ingress1024_argmax]]
set part "xc7z020clg400-1"
set clk 10
set project_name "seq256_ingress1024_dsp_margin_argmax"
puts "== SEQ256 DSP-margin + on-FPGA argmax: part=$part clk=${clk}ns =="
open_project -reset $project_name
set_top transformer_top
add_files [file join $here transformer_top_seq256_ingress1024_dsp_margin_argmax.cpp] \
    -cflags "-DHLS_USE_AP_INT -I$here"
add_files -tb [file join $here testbench_ingress1024_argmax.cpp]
add_files -tb $golden
open_solution -reset sol1 -flow_target vivado
set_part $part
create_clock -period $clk -name default
csim_design -O -argv $golden
csynth_design
export_design -format ip_catalog
exit
