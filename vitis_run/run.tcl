open_project spmmSketch
set_top spgemm
add_files ../spmmSketch/hls/common.h
add_files ../spmmSketch/hls/spgemm.cpp
add_files -tb ../spmmSketch/hls/spgemm-tb.cpp
open_solution "solution1" -flow_target vivado
set_part {xcvu11p-flga2577-1-e}
create_clock -period 10 -name default

csim_design
csynth_design
cosim_design
export_design -format ip_catalog
