set here [file dirname [file normalize [info script]]]
set golden [file normalize [file join $here .. golden]]
set part "xc7z020clg400-1"
if {[info exists ::env(FPGA_PART)] && $::env(FPGA_PART) ne ""} { set part $::env(FPGA_PART) }
set clk 10
if {[info exists ::env(CLOCK_NS)] && $::env(CLOCK_NS) ne ""} { set clk $::env(CLOCK_NS) }
set project_name "seq256_int10_[string map {- _} $part]"
puts "== SEQ256 int10 run: part=$part clk=${clk}ns =="
open_project -reset $project_name
set_top transformer_top
add_files [file join $here transformer_top_seq256.cpp] -cflags "-DHLS_USE_AP_INT -I$here"
add_files -tb [file join $here testbench_top.cpp]
add_files -tb $golden
open_solution -reset sol1 -flow_target vivado
set_part $part
create_clock -period $clk -name default
puts "== CSIM start =="
csim_design -O -argv $golden
puts "== CSIM done; CSYNTH start =="
csynth_design
puts "== CSYNTH done =="
exit
