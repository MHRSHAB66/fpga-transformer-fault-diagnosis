# MYIR Z-Turn XC7Z020 integration: Zynq PS + AXI DMA + SEQ=256 HLS IP.
set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir zturn_7z020_ps_config.tcl]
set root [file normalize [file join $script_dir .. .. ..]]
set build_dir [file normalize [file join $root tmp seq256_dma_system]]
set ip_repo [file normalize [file join $root tmp seq256_ingress1024_run seq256_ingress1024_int10_xc7z020clg400_1 sol1 impl ip]]
set part "xc7z020clg400-1"

file mkdir $build_dir
create_project -force seq256_dma_system $build_dir -part $part
set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog

create_bd_design "design_1"
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:* processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "0"} \
    [get_bd_cells processing_system7_0]
configure_zturn_7z020_ps [get_bd_cells processing_system7_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* axi_dma_0
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_s_axis_s2mm_tdata_width {32} \
    CONFIG.c_include_mm2s_dre {0} \
    CONFIG.c_include_s2mm_dre {0}] [get_bd_cells axi_dma_0]

create_bd_cell -type ip -vlnv xilinx.com:hls:transformer_top:1.0 transformer_top_0

# PS master controls both AXI-Lite slaves.
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/processing_system7_0/M_AXI_GP0" Clk "Auto"} \
    [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/processing_system7_0/M_AXI_GP0" Clk "Auto"} \
    [get_bd_intf_pins transformer_top_0/s_axi_ctrl]

# DMA memory traffic shares the PS high-performance slave port through a
# two-input AXI interconnect. This is explicit because Vivado 2020.2's AXI
# automation does not reliably discover a newly enabled generic-part HP port.
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:* axi_mem_interconnect
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] \
    [get_bd_cells axi_mem_interconnect]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] \
    [get_bd_intf_pins axi_mem_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] \
    [get_bd_intf_pins axi_mem_interconnect/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_mem_interconnect/M00_AXI] \
    [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] \
    [get_bd_intf_pins transformer_top_0/in_r]
connect_bd_intf_net [get_bd_intf_pins transformer_top_0/out_r] \
    [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# AXI-Lite automation already connects the HLS clock/reset. Extend the same
# 100 MHz fabric domain to the explicit HP0 memory interconnect.
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] \
    [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
    [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] \
    [get_bd_pins axi_mem_interconnect/ACLK] \
    [get_bd_pins axi_mem_interconnect/S00_ACLK] \
    [get_bd_pins axi_mem_interconnect/S01_ACLK] \
    [get_bd_pins axi_mem_interconnect/M00_ACLK]
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] \
    [get_bd_pins axi_mem_interconnect/ARESETN] \
    [get_bd_pins axi_mem_interconnect/S00_ARESETN] \
    [get_bd_pins axi_mem_interconnect/S01_ARESETN] \
    [get_bd_pins axi_mem_interconnect/M00_ARESETN]

assign_bd_address
validate_bd_design
save_bd_design

set bd_file [get_files design_1.bd]
generate_target all $bd_file
make_wrapper -files $bd_file -top
add_files -norecurse [file join $build_dir seq256_dma_system.gen sources_1 bd design_1 hdl design_1_wrapper.v]
set_property top design_1_wrapper [current_fileset]

launch_runs synth_1 -jobs 4
wait_on_run synth_1
open_run synth_1
report_utilization -file [file join $build_dir post_synth_utilization.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
    -file [file join $build_dir post_synth_timing_summary.rpt]

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
open_run impl_1
report_utilization -file [file join $build_dir post_route_utilization.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
    -file [file join $build_dir post_route_timing_summary.rpt]
report_route_status -file [file join $build_dir post_route_status.rpt]
report_drc -file [file join $build_dir post_route_drc.rpt]
write_hw_platform -fixed -include_bit -force -file [file join $build_dir seq256_dma_system.xsa]
