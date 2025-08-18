# ============================================================================
# Vivado Project Creation Script for CNN Accelerator
# VR Hand Gesture Recognition for Zybo Z7-20
# 
# Engineer: Sadad Haidari
# Target: XC7Z020CLG400-1 (Zybo Z7-20)
# ============================================================================

# Project settings
set project_name "cnn_accelerator_rtl"
set project_dir [pwd]
set target_device "xc7z020clg400-1"

puts "=============================================="
puts "CREATING VIVADO PROJECT FOR CNN ACCELERATOR"
puts "=============================================="
puts "Project: $project_name"
puts "Target: $target_device"
puts "Directory: $project_dir"

# Remove existing project
if {[file exists $project_name]} {
    puts "Removing existing project..."
    file delete -force $project_name
}

# Create project
puts "Creating Vivado project..."
create_project $project_name $project_dir -part $target_device -force

# Set project properties
set_property target_language Verilog [current_project]

# Try to set board part, but don't fail if not available
if {[catch {set_property board_part digilentinc.com:zybo-z7-20:part0:1.0 [current_project]} result]} {
    puts "Warning: Board part not found, continuing without board-specific settings"
    puts "Result: $result"
}

# Add RTL source files
puts "Adding RTL source files..."
add_files -norecurse [list \
    "cnn_accelerator.v" \
    "conv_layer.v" \
]

# Add constraint files
puts "Adding constraint files..."
add_files -fileset constrs_1 -norecurse [list \
    "cnn_accelerator.xdc" \
]

# Create IP Integrator block design
puts "Creating IP Integrator block design..."
create_bd_design "cnn_accelerator_bd"

# Add Zynq Processing System
puts "Adding Zynq Processing System..."
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0

# Configure Zynq PS
set_property -dict [list CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {200} \
                        CONFIG.PCW_USE_S_AXI_HP0 {1} \
                        CONFIG.PCW_USE_S_AXI_GP0 {1} \
                        CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
                        CONFIG.PCW_IRQ_F2P_INTR {1}] [get_bd_cells processing_system7_0]

# Add AXI Interconnect for control interface
puts "Adding AXI Interconnect for control..."
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0

# Configure AXI Interconnect for control
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_0]

# Add AXI Interconnect for memory interface
puts "Adding AXI Interconnect for memory..."
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_1

# Configure AXI Interconnect for memory
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_1]

# Add CNN Accelerator IP
puts "Adding CNN Accelerator IP..."
create_bd_cell -type module -reference cnn_accelerator cnn_accelerator_0

# Connect interfaces
puts "Connecting interfaces..."

# Connect clock and reset
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins cnn_accelerator_0/clk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins cnn_accelerator_0/rst_n]

# Connect AXI Interconnect clocks and resets
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins axi_interconnect_0/ARESETN]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_1/ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins axi_interconnect_1/ARESETN]

# Connect AXI interface clocks
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/S_AXI_GP0_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]

# Connect AXI Interconnect interface clocks
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/S00_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/M00_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_1/S00_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_1/M00_ACLK]

# Connect AXI-Lite interface through interconnect
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins axi_interconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M00_AXI] [get_bd_intf_pins cnn_accelerator_0/s_axil]

# Connect AXI-MM interface through interconnect
connect_bd_intf_net [get_bd_intf_pins cnn_accelerator_0/m_axi] [get_bd_intf_pins axi_interconnect_1/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_1/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

# Auto-assign addresses
assign_bd_address

# Save the block design
save_bd_design

# Generate wrapper
puts "Generating wrapper..."
make_wrapper -files [get_files cnn_accelerator_bd.bd] -top

# Add the generated wrapper file from the .gen location
add_files -norecurse ${project_dir}/${project_name}.gen/sources_1/bd/cnn_accelerator_bd/hdl/cnn_accelerator_bd_wrapper.v

# Set top module
set_property top cnn_accelerator_bd_wrapper [current_fileset]
set_property top_file ${project_dir}/${project_name}.gen/sources_1/bd/cnn_accelerator_bd/hdl/cnn_accelerator_bd_wrapper.v [current_fileset]

# Generate bitstream
puts "Generating bitstream..."
launch_runs synth_1 -jobs 4
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] == "100%"} {
    puts "✅ Synthesis completed successfully"
    
    launch_runs impl_1 -to_step write_bitstream -jobs 4
    wait_on_run impl_1
    
    if {[get_property PROGRESS [get_runs impl_1]] == "100%"} {
        puts "✅ Implementation completed successfully"
        puts "✅ Bitstream generated: ${project_name}.bit"
    } else {
        puts "❌ Implementation failed"
    }
} else {
    puts "❌ Synthesis failed"
}

puts ""
puts "=============================================="
puts "PROJECT CREATION COMPLETED"
puts "=============================================="
puts "Project location: $project_dir/$project_name"
puts "Bitstream: $project_dir/$project_name/$project_name.runs/impl_1/cnn_accelerator_bd_wrapper.bit"
puts ""
puts "Next steps:"
puts "1. Open Vivado IDE: vivado $project_name.xpr"
puts "2. Review synthesis and implementation reports"
puts "3. Program FPGA with bitstream"
puts "4. Test CNN accelerator functionality"

close_project
