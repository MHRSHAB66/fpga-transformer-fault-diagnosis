# Re-run only the implementation of an existing system project with another
# Vivado strategy (synthesis, block design and IP are reused unchanged), then
# write the same reports and XSA as the build scripts.
#   vivado -mode batch -notrace -source rerun_impl_strategy.tcl \
#       -tclargs <project.xpr> <strategy>
# e.g. -tclargs tmp/vvote/vvote.xpr Performance_ExplorePostRoutePhysOpt
set xpr [file normalize [lindex $argv 0]]
set strategy [lindex $argv 1]
set build_dir [file dirname $xpr]

open_project $xpr
reset_run impl_1
set_property strategy $strategy [get_runs impl_1]
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1
open_run impl_1
report_utilization -file [file join $build_dir post_route_utilization.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
    -file [file join $build_dir post_route_timing_summary.rpt]
report_route_status -file [file join $build_dir post_route_status.rpt]
report_drc -file [file join $build_dir post_route_drc.rpt]
write_hw_platform -fixed -include_bit -force -file [file join $build_dir seq256_dma_system.xsa]
puts "IMPL_STRATEGY $strategy"
