# HLS CNN Accelerator Build Script
# Generated for Zybo Z7-20 FPGA

# Create new project
open_project cnn_accelerator_hls
set_top cnn_accelerator

# Add source files
add_files ../hls_src/cnn_accelerator.cpp
add_files ../hls_headers/layer_config.h
add_files ../hls_headers/bram_layers.h
add_files ../hls_headers/ddr_layers.h
add_files ../hls_headers/streaming_layers.h
add_files ../weights.h

# Add testbench
add_files -tb ../testbench/cnn_testbench.cpp
add_files -tb ../testbench/test_input.h

# Create solution
open_solution "solution1"
set_part {xc7z020clg400-1}  # Zybo Z7-20
create_clock -period 5 -name default  # 200MHz

# Synthesis
csynth_design

# Co-simulation (optional)
# cosim_design

# Export IP
export_design -format ip_catalog

exit
