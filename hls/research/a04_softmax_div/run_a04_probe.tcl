# A04 single-operator probe: canonical division vs 9-bit-quotient divider.
# One project per variant; each runs C-synthesis; ooc_impl.tcl then runs an out-of-context Vivado
# implementation of the generated RTL for real LUT/FF/DSP/timing.
# Usage (run from tmp/ so the projects land in the gitignored tmp/):
#   cd tmp && vitis_hls -f ../hls/research/a04_softmax_div/run_a04_probe.tcl
set here [file dirname [file normalize [info script]]]
set vitis [file normalize [file join $here .. .. f1_int10_seq256 vitis]]
foreach {variant flags} [list orig "" divq9 "-DA04_DIVQ9"] {
    open_project -reset a04_probe_$variant
    set_top a04_probe
    add_files [file join $here a04_probe.cpp] -cflags "-DHLS_USE_AP_INT -I$vitis -I$here $flags"
    open_solution -reset sol1 -flow_target vivado
    set_part xc7z020clg400-1
    create_clock -period 10 -name default
    csynth_design
    # IP export is skipped (Vitis HLS 2020.2 date-revision packaging bug);
    # the RTL is implemented out of context by ooc_impl.tcl instead.
    close_project
}
exit
