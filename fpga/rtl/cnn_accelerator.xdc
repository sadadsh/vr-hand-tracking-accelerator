# ============================================================================
# XDC Constraints for CNN Accelerator RTL
# VR Hand Gesture Recognition for Zybo Z7-20
# 
# Engineer: Sadad Haidari
# Target: XC7Z020CLG400-1 (Zybo Z7-20)
# ============================================================================

# ============================================================================
# CLOCK CONSTRAINTS
# ============================================================================

# System clock (200MHz from Zynq PS)
create_clock -period 5.000 -name clk -waveform {0.000 2.500} [get_ports clk]

# Clock uncertainty
set_clock_uncertainty 0.500 [get_clocks clk]

# ============================================================================
# TIMING CONSTRAINTS
# ============================================================================

# Input delay constraints
set_input_delay -clock clk -max 1.000 [get_ports rst_n]
set_input_delay -clock clk -max 1.000 [get_ports s_axil_*]

# Output delay constraints
set_output_delay -clock clk -max 1.000 [get_ports m_axi_*]

# ============================================================================
# AXI INTERFACE CONSTRAINTS
# ============================================================================

# AXI-Lite slave interface
set_property INTERFACE_MODE slave [get_bd_intf_pins cnn_accelerator_0/s_axil]
set_property CONFIG.ADDR_WIDTH {32} [get_bd_intf_pins cnn_accelerator_0/s_axil]
set_property CONFIG.DATA_WIDTH {32} [get_bd_intf_pins cnn_accelerator_0/s_axil]

# AXI-MM master interface
set_property INTERFACE_MODE master [get_bd_intf_pins cnn_accelerator_0/m_axi]
set_property CONFIG.ADDR_WIDTH {32} [get_bd_intf_pins cnn_accelerator_0/m_axi]
set_property CONFIG.DATA_WIDTH {512} [get_bd_intf_pins cnn_accelerator_0/m_axi]
set_property CONFIG.ID_WIDTH {0} [get_bd_intf_pins cnn_accelerator_0/m_axi]

# ============================================================================
# MEMORY CONSTRAINTS
# ============================================================================

# BRAM constraints for feature maps
set_property RAM_STYLE block [get_cells feature_map_a*]
set_property RAM_STYLE block [get_cells feature_map_b*]

# Weight buffer constraints
set_property RAM_STYLE distributed [get_cells weight_buffer*]

# ============================================================================
# PERFORMANCE CONSTRAINTS
# ============================================================================

# Pipeline constraints for convolution
set_false_path -from [get_pins */conv_layer_*/state_reg*/C] -to [get_pins */conv_layer_*/next_state_reg*/D]

# Multiplier constraints
set_property USE_DSP yes [get_cells mult_results*]

# ============================================================================
# AREA CONSTRAINTS
# ============================================================================

# Limit resource usage for Zybo Z7-20
# BRAM: 280 blocks (use max 70% = 196 blocks)
# DSP: 220 slices (use max 80% = 176 slices)
# LUT: 53200 (use max 60% = 31920)

set_property CONFIG.BRAM_PORTA_WIDTH {8} [get_bd_cells cnn_accelerator_0]
set_property CONFIG.BRAM_PORTB_WIDTH {8} [get_bd_cells cnn_accelerator_0]

# ============================================================================
# POWER CONSTRAINTS
# ============================================================================

# Power optimization
set_property CONFIG.POWER_OPTIMIZATION true [current_design]

# ============================================================================
# DEBUG CONSTRAINTS
# ============================================================================

# Debug signals (if using ChipScope)
# set_property MARK_DEBUG true [get_nets processing_done]
# set_property MARK_DEBUG true [get_nets state]
# set_property MARK_DEBUG true [get_nets layer_counter]

# ============================================================================
# PHYSICAL CONSTRAINTS
# ============================================================================

# Pin assignments (if needed for external interfaces)
# set_property PACKAGE_PIN P16 [get_ports clk]
# set_property PACKAGE_PIN R19 [get_ports rst_n]

# ============================================================================
# SYNTHESIS CONSTRAINTS
# ============================================================================

# Synthesis optimization
set_property STEPS.SYNTH_DESIGN.TCL.PRE {} [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.TCL.POST {} [get_runs synth_1]

# Implementation optimization
set_property STEPS.OPT_DESIGN.TCL.PRE {} [get_runs impl_1]
set_property STEPS.OPT_DESIGN.TCL.POST {} [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.TCL.PRE {} [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.TCL.POST {} [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.TCL.PRE {} [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.TCL.POST {} [get_runs impl_1]

# ============================================================================
# VERIFICATION CONSTRAINTS
# ============================================================================

# Simulation constraints
set_property SIMULATE_MODEL_TYPOLOGY post_synthesis [current_fileset -simset]
set_property SIMULATE_BEHAVIORAL_LIBRARY xil_defaultlib [current_fileset -simset]

# ============================================================================
# END OF CONSTRAINTS
# ============================================================================
