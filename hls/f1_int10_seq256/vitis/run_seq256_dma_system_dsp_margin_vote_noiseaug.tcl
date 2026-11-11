# NOISE-AUGMENTED weights on the RAW-INPUT + TEMPORAL VOTE core (2026-11-11), from the raw-input system. Identical system to
# run_seq256_dma_system_dsp_margin_argmax.tcl (dma16: one 32,768-byte MM2S frame,
# no SG, transformer IP synthesized with MAX_DSP 210) except that it packages the
# IP from seq256_rawvote1024 (preprocessing + temporal vote inside the IP) and
# builds into its own directory.
# MYIR Z-Turn XC7Z020 integration: Zynq PS + AXI DMA + SEQ=256 HLS IP.
set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir zturn_7z020_ps_config.tcl]
set root [file normalize [file join $script_dir .. .. ..]]
# Short build dir/project name: the HLS module file names are long and the
# Windows 260-character path limit was exceeded with the descriptive name.
set proj_name vna
set build_dir [file normalize [file join $root tmp $proj_name]]
set ip_repo [file normalize [file join $root seq256_rawvote1024_na sol1 impl ip]]
set part "xc7z020clg400-1"

file mkdir $build_dir
create_project -force $proj_name $build_dir -part $part
set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog

create_bd_design "design_1"
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:* processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "0"} \
    [get_bd_cells processing_system7_0]
configure_zturn_7z020_ps [get_bd_cells processing_system7_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* axi_dma_0
# Direct-register (simple) mode. The buffer-length register defaults to 14 bits
# (max 16384 B/transfer), but one input frame is 8192 x 32-bit = 32768 B, which
# must be sent as a SINGLE MM2S transfer so TLAST falls on word 8191 (the HLS
# core reads one 8192-word frame). Widen the length register to 16 bits
# (max 65536 B) WITHOUT enabling Scatter-Gather. 32-bit stream widths, instance
# names and AXI-Lite base addresses are unchanged.
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {16} \
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
add_files -norecurse [file join $build_dir $proj_name.gen sources_1 bd design_1 hdl design_1_wrapper.v]
set_property top design_1_wrapper [current_fileset]

# Keep ten DSP48 sites free at the implemented-system level. HLS resource
# estimates count operators, whereas Vivado can expand wide multipliers into
# additional DSP48 primitives during OOC synthesis. Limit the Transformer IP
# synthesis itself so the margin is real rather than only an HLS estimate.
create_ip_run $bd_file
set transformer_synth_run [get_runs design_1_transformer_top_0_0_synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.MAX_DSP 210 $transformer_synth_run

launch_runs synth_1 -jobs 2
wait_on_run synth_1
open_run synth_1
report_utilization -file [file join $build_dir post_synth_utilization.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
    -file [file join $build_dir post_synth_timing_summary.rpt]

launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1
open_run impl_1
report_utilization -file [file join $build_dir post_route_utilization.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
    -file [file join $build_dir post_route_timing_summary.rpt]
report_route_status -file [file join $build_dir post_route_status.rpt]
report_drc -file [file join $build_dir post_route_drc.rpt]
write_hw_platform -fixed -include_bit -force -file [file join $build_dir seq256_dma_system.xsa]
