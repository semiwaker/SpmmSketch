open_project spGEMM
set_top spgemm
add_files ../spmmSketch/hls/common.h
add_files ../spmmSketch/hls/spgemm.cpp
add_files -tb ../spmmSketch/hls/spgemm-tb.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1" -flow_target vivado
set_part {xcvu9p-flga2104-1-e}
create_clock -period 4 -name default
config_interface -m_axi_alignment_byte_size 64 -m_axi_latency 64 -m_axi_max_widen_bitwidth 512
config_rtl -register_reset_num 3

csim_design -ldflags {-Wl,-z,stack-size=104857600} -clean
csynth_design
#cosim_design -ldflags {-Wl,--stack,10485760}
#export_design -format ip_catalog
