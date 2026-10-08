set project_name "seq256_ingress1024_int10_xc7z020clg400_1"
puts "== Opening completed HLS project: $project_name =="
open_project $project_name
open_solution sol1
puts "== Vivado implementation start =="
export_design -flow impl -rtl verilog -format ip_catalog
puts "== Vivado implementation done =="
exit
