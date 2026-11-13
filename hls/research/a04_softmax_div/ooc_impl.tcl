# Out-of-context Vivado synthesis + place + route of an HLS-generated RTL
# directory (research probes; no IP packaging needed).
#   vivado -mode batch -notrace -source ooc_impl.tcl -tclargs <verilog_dir> <top> <out_dir> [clock_ns]
set rtl [lindex $argv 0]
set top [lindex $argv 1]
set out [lindex $argv 2]
set period [expr {[llength $argv] > 3 ? [lindex $argv 3] : 10.0}]
file mkdir $out
read_verilog [glob [file join $rtl *.v]]
foreach f [glob -nocomplain [file join $rtl *.dat]] { file copy -force $f [pwd] }
synth_design -top $top -part xc7z020clg400-1 -mode out_of_context
create_clock -period $period -name ap_clk [get_ports ap_clk]
opt_design
place_design
route_design
report_utilization -file [file join $out post_route_utilization.rpt]
report_timing_summary -file [file join $out post_route_timing_summary.rpt]
puts "OOC_DONE $top"
