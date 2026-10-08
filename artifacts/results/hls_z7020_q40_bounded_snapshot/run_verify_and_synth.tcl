set here [file dirname [file normalize [info script]]]
set golden [file join $here golden]
open_project verified_q40_z7020
set_top transformer_top
add_files [file join $here transformer_top.cpp] -cflags "-DHLS_USE_AP_INT -I$here"
add_files -tb [file join $here testbench_top.cpp]
add_files -tb $golden
open_solution sol1 -flow_target vivado
set_part xc7z020clg400-1
create_clock -period 10 -name default
csim_design -O -argv $golden
csynth_design
exit
