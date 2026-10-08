set script_dir [file dirname [file normalize [info script]]]
set root [file normalize [file join $script_dir .. .. ..]]
set build_dir [file normalize [file join $root tmp seq256_dma_system_dsp_margin]]
set project_file [file join $build_dir seq256_dma_system_dsp_margin.xpr]

open_project $project_file

# The previous process stopped during placement. Reuse the completed synthesis
# checkpoint and restart only implementation through bitstream generation.
reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "impl_1 did not complete: [get_property STATUS [get_runs impl_1]]"
}

open_run impl_1
report_utilization -file [file join $build_dir post_route_utilization.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
    -file [file join $build_dir post_route_timing_summary.rpt]
report_route_status -file [file join $build_dir post_route_status.rpt]
report_drc -file [file join $build_dir post_route_drc.rpt]
write_hw_platform -fixed -include_bit -force \
    -file [file join $build_dir seq256_dma_system.xsa]

close_project
exit
