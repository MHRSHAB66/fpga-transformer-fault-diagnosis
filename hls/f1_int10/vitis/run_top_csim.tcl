# C-simulate the actual AXI-Stream synthesis top against exported golden data.
# Run from an ignored scratch directory, e.g. repo/tmp/hls_top_csim.
# No device is selected here: this is functional C simulation, not synthesis.
set here [file dirname [file normalize [info script]]]
set golden [file normalize [file join $here .. golden]]

open_project fault_int10_top_csim
set_top transformer_top
add_files -cflags "-DHLS_USE_AP_INT" [file join $here transformer_top.cpp]
add_files -tb [file join $here testbench_top.cpp]
add_files -tb $golden
open_solution sol1 -flow_target vivado
csim_design -O -argv $golden
exit
