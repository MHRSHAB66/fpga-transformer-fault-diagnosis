# NOISE-AUGMENTED model on the raw-input + temporal vote core (2026-11-11, paper
# study; the deployed model and its sources are untouched). Same top and headers
# as run_seq256_rawvote1024_verify_and_synth.tcl; only weights_int10.h differs:
# it is generated from artifacts/quantized/f_seq256_int10_noiseaug/. The sources
# are copied into tmp/noiseaug_vitis_src/ because the top includes
# "weights_int10.h" from its own directory.
# C-simulation: the variant golden set (same 41 raw windows, expected logits from
# the variant's INT10 reference) as one stream with vote_k = 5.
# Usage: vitis_hls -f run_seq256_rawvote1024_noiseaug_verify_and_synth.tcl
set here [file dirname [file normalize [info script]]]
set root [file normalize [file join $here .. .. ..]]
set variant [file join $root artifacts quantized f_seq256_int10_noiseaug]
set golden [file join $variant golden_raw1024_argmax]
set src [file join $root tmp noiseaug_vitis_src]
file delete -force $src
file mkdir $src
foreach f [glob [file join $here *.h]] { file copy -force $f $src }
file copy -force [file join $here transformer_top_seq256_raw1024_dsp_margin_vote.cpp] $src
file copy -force [file join $variant weights_int10.h] [file join $src weights_int10.h]
set part "xc7z020clg400-1"
set clk 10
set project_name "seq256_rawvote1024_na"
puts "== SEQ256 raw-input + vote core, NOISE-AUGMENTED weights: part=$part clk=${clk}ns =="
open_project -reset $project_name
set_top transformer_top
add_files [file join $src transformer_top_seq256_raw1024_dsp_margin_vote.cpp] \
    -cflags "-DHLS_USE_AP_INT -I$src"
add_files -tb [file join $here testbench_raw1024_vote.cpp] -cflags "-I$src"
add_files -tb $golden
open_solution -reset sol1 -flow_target vivado
set_part $part
create_clock -period $clk -name default
csim_design -O -argv "$golden 5"
csynth_design
export_design -format ip_catalog
exit
