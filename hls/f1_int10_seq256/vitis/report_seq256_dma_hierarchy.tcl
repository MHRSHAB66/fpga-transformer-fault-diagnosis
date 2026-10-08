set root [file normalize [file join [file dirname [file normalize [info script]]] .. .. ..]]
set dcp [file join $root tmp seq256_dma_system seq256_dma_system.runs impl_1 design_1_wrapper_routed.dcp]
set out [file join $root tmp seq256_dma_system post_route_hierarchical_utilization.rpt]
open_checkpoint $dcp
report_utilization -hierarchical -hierarchical_depth 8 -file $out
