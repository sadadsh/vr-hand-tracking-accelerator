# ============================================================================
# XDC Constraints for CNN Accelerator RTL
# VR Hand Gesture Recognition for Zybo Z7-20
# 
# Engineer: Sadad Haidari
# Target: XC7Z020CLG400-1 (Zybo Z7-20)
# 
# Note: This file contains XDC constraints only.
# Block design configuration is in create_vivado_project.tcl
# Synthesis/implementation hooks are in the TCL script as well.
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

# Input delay constraints (for external interfaces if any)
# set_input_delay -clock clk -max 1.000 [get_ports rst_n]

# Note: AXI interface timing is handled automatically by Vivado
# when using block design with proper AXI IP cores

# ============================================================================
# AXI INTERFACE CONSTRAINTS
# ============================================================================

# Note: AXI interface configuration is handled in the block design TCL script
# These constraints focus on timing and physical aspects only

# ============================================================================
# MEMORY CONSTRAINTS
# ============================================================================

# BRAM constraints for feature maps (large storage)
set_property RAM_STYLE block [get_cells feature_map_a*]
set_property RAM_STYLE block [get_cells feature_map_b*]

# Weight buffer constraints (small, fast access)
set_property RAM_STYLE distributed [get_cells weight_buffer*]

# CNN-specific memory optimizations
# Force BRAM for large feature maps (adjust names based on actual RTL)
# set_property RAM_STYLE block [get_cells conv_layer_*/feature_map*]
# set_property RAM_STYLE block [get_cells conv_layer_*/output_buffer*]

# Force distributed RAM for small weight caches (adjust names based on actual RTL)
# set_property RAM_STYLE distributed [get_cells conv_layer_*/weight_cache*]
# set_property RAM_STYLE distributed [get_cells conv_layer_*/bias_buffer*]

# ============================================================================
# PERFORMANCE CONSTRAINTS
# ============================================================================

# CNN-specific timing constraints
# False paths for state machine transitions
set_false_path -from [get_pins */conv_layer_*/state_reg*/C] -to [get_pins */conv_layer_*/next_state_reg*/D]

# False paths for control signals
set_false_path -from [get_pins */start] -to [get_pins */done]
set_false_path -from [get_pins */input_valid] -to [get_pins */output_valid]

# Multiplier constraints - force DSP usage for efficiency
set_property USE_DSP yes [get_cells mult_results*]
set_property USE_DSP yes [get_cells conv_layer_*/mult*]

# Multicycle paths for CNN computations (relax timing for complex ops)
# Note: Adjust signal names based on actual RTL implementation
# set_multicycle_path -setup 2 -from [get_clocks clk] -to [get_clocks clk] -through [get_pins conv_layer_*/conv_compute*]
# set_multicycle_path -hold 1 -from [get_clocks clk] -to [get_clocks clk] -through [get_pins conv_layer_*/conv_compute*]

# ============================================================================
# AREA & PLACEMENT CONSTRAINTS
# ============================================================================

# Resource limits for Zybo Z7-20
# BRAM: 280 blocks (use max 70% = 196 blocks)
# DSP: 220 slices (use max 80% = 176 slices)
# LUT: 53200 (use max 60% = 31920)

# CNN-specific placement constraints (optional - uncomment if needed)
# Keep convolution layers close together for better timing
# set_property LOC SLICE_X0Y0 [get_cells conv_layer_*]
# set_property LOC SLICE_X0Y1 [get_cells conv_layer_*/conv_compute*]

# Keep memory blocks near their consumers
# set_property LOC RAMB18_X0Y0 [get_cells feature_map_a*]
# set_property LOC RAMB18_X0Y1 [get_cells feature_map_b*]

# Note: Block design configuration is handled in create_vivado_project.tcl
# These constraints focus on synthesis and implementation optimization

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
# SYNTHESIS & IMPLEMENTATION CONSTRAINTS
# ============================================================================

# Note: Synthesis and implementation TCL hooks are configured in
# create_vivado_project.tcl for better project management

# ============================================================================
# VERIFICATION CONSTRAINTS
# ============================================================================

# Simulation constraints
set_property SIMULATE_MODEL_TYPOLOGY post_synthesis [current_fileset -simset]
set_property SIMULATE_BEHAVIORAL_LIBRARY xil_defaultlib [current_fileset -simset]

# ============================================================================
# OPTIONAL CONSTRAINTS (Uncomment as needed)
# ============================================================================

# Use these constraints when you need:
# 1. Higher clock frequencies (>200MHz)
# 2. Better timing margins (current slack issues)
# 3. Lower power consumption
# 4. Resource optimization (hitting utilization limits)
# 5. Specific placement for performance optimization

# To enable specific constraints:
# 1. Uncomment the desired constraint lines above
# 2. Adjust signal names to match your actual RTL implementation
# 3. Verify placement locations work with your floorplan
# 4. Re-run synthesis and implementation

# ============================================================================
# END OF CONSTRAINTS
# ============================================================================
