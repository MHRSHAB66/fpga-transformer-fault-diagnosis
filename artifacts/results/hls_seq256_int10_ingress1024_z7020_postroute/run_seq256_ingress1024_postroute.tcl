set part "xc7z020clg400-1"
set root [file normalize [file join [file dirname [info script]] .. .. ..]]
set generated [file normalize [file join $root tmp seq256_ingress1024_run seq256_ingress1024_int10_xc7z020clg400_1 sol1 impl ip]]
set out_dir [file normalize [file join $root tmp seq256_ingress1024_postroute]]
file mkdir $out_dir

puts "== RTL: $generated/hdl/verilog =="
puts "== post-route output: $out_dir =="
read_verilog [glob [file join $generated hdl verilog *.v]]
read_xdc [file join $generated constraints transformer_top_ooc.xdc]

synth_design -top transformer_top -part $part -mode out_of_context
write_checkpoint -force [file join $out_dir post_synth.dcp]
report_utilization -file [file join $out_dir post_synth_utilization.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -file [file join $out_dir post_synth_timing_summary.rpt]

opt_design
place_design
phys_opt_design
write_checkpoint -force [file join $out_dir post_place.dcp]
report_utilization -file [file join $out_dir post_place_utilization.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -file [file join $out_dir post_place_timing_summary.rpt]

route_design
write_checkpoint -force [file join $out_dir post_route.dcp]
report_utilization -file [file join $out_dir post_route_utilization.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -file [file join $out_dir post_route_timing_summary.rpt]
report_route_status -file [file join $out_dir post_route_status.rpt]
report_drc -file [file join $out_dir post_route_drc.rpt]
puts "== post-route analysis complete =="
