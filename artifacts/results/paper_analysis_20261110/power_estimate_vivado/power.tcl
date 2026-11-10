# Vectorless Vivado power estimate of the two routed system checkpoints.
foreach {name dcp} {
  raw_argmax   G:/aut/project/fpga_fault_diagnosis/tmp/vraw/vraw.runs/impl_1/design_1_wrapper_routed.dcp
  int10_argmax G:/aut/project/fpga_fault_diagnosis/tmp/seq256_dma_system_dsp_margin_argmax/seq256_dma_system_dsp_margin_argmax.runs/impl_1/design_1_wrapper_routed.dcp
} {
  open_checkpoint $dcp
  report_power -file G:/aut/project/fpga_fault_diagnosis/tmp/power_est/power_${name}.rpt
  report_power -hierarchical_depth 4 -file G:/aut/project/fpga_fault_diagnosis/tmp/power_est/power_${name}_hier.rpt
  close_design
}
