set script_dir [file dirname [file normalize [info script]]]
set root [file normalize [file join $script_dir .. .. ..]]
set build_dir [file normalize [file join $root tmp validate_zturn_7z020_ps]]

source [file join $script_dir zturn_7z020_ps_config.tcl]
create_project -force validate_zturn_7z020_ps $build_dir -part xc7z020clg400-1
create_bd_design design_1
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:* processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "0"} \
    [get_bd_cells processing_system7_0]
configure_zturn_7z020_ps [get_bd_cells processing_system7_0]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] \
    [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]
validate_bd_design
save_bd_design

set ps [get_bd_cells processing_system7_0]
foreach property_name {
    CONFIG.PCW_DDR_RAM_HIGHADDR
    CONFIG.PCW_UIPARAM_DDR_PARTNO
    CONFIG.PCW_QSPI_PERIPHERAL_ENABLE
    CONFIG.PCW_ENET0_PERIPHERAL_ENABLE
    CONFIG.PCW_SD0_PERIPHERAL_ENABLE
    CONFIG.PCW_UART1_PERIPHERAL_ENABLE
    CONFIG.PCW_USB0_PERIPHERAL_ENABLE
    CONFIG.PCW_USE_S_AXI_HP0
} {
    puts "ZTURN_CHECK $property_name=[get_property $property_name $ps]"
}
close_project
