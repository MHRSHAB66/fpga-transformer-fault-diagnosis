set script_dir [file dirname [file normalize [info script]]]
set root [file normalize [file join $script_dir .. .. ..]]
set build_dir [file normalize [file join $root tmp seq256_dma_system_dsp_margin]]
open_project [file join $build_dir seq256_dma_system_dsp_margin.xpr]
open_run synth_1

set out [open [file join $build_dir dsp_cell_inventory.txt] w]
foreach cell [get_cells -hierarchical -filter {REF_NAME =~ DSP48*}] {
    puts $out [get_property NAME $cell]
}
close $out
close_project
exit
