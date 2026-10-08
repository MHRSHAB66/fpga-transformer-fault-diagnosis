set here [file dirname [file normalize [info script]]]
set golden [file normalize [file join $here .. golden_ingress1024]]
set part "xc7z020clg400-1"
set clk 10
set project_name "seq256_ingress1024_dsp_margin"
puts "== SEQ256 DSP-margin classifier variant: part=$part clk=${clk}ns =="
open_project -reset $project_name
set_top transformer_top
add_files [file join $here transformer_top_seq256_ingress1024_dsp_margin.cpp] \
    -cflags "-DHLS_USE_AP_INT -I$here"
add_files -tb [file join $here testbench_ingress1024.cpp]
add_files -tb $golden
open_solution -reset sol1 -flow_target vivado
set_part $part
create_clock -period $clk -name default
csim_design -O -argv $golden
csynth_design
exit
