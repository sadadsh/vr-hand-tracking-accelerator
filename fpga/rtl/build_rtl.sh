#!/bin/bash

# ============================================================================
# CNN Accelerator RTL Build Script
# VR Hand Gesture Recognition for Zybo Z7-20
# 
# Engineer: Sadad Haidari
# Target: XC7Z020CLG400-1 (Zybo Z7-20)
# ============================================================================

echo "=============================================="
echo "CNN ACCELERATOR RTL BUILD"
echo "=============================================="
echo "Target: Zybo Z7-20 (XC7Z020CLG400-1)"
echo "Architecture: INT4 weights, INT8 activations"
echo "Model: 319 layers, 89.31% accuracy"
echo ""

# Check if we're in the right directory
if [ ! -f "cnn_accelerator.v" ]; then
    echo "❌ Error: cnn_accelerator.v not found!"
    echo "Please run this script from the fpga/rtl/ directory"
    exit 1
fi

# Source Vivado environment
echo "Sourcing Vivado 2025.1 environment..."
source /tools/Xilinx/2025.1/Vivado/.settings64-Vivado.sh

# Check for Vivado
if command -v vivado &> /dev/null; then
    echo "✅ Found Vivado command"
else
    echo "❌ Error: Vivado not found!"
    echo "Please ensure Vivado 2025.1 is properly installed"
    exit 1
fi

# Check for required files
echo ""
echo "Checking required files..."
REQUIRED_FILES=(
    "cnn_accelerator.v"
    "conv_layer.v"
    "cnn_accelerator.xdc"
    "create_vivado_project.tcl"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (missing)"
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo ""
    echo "❌ Missing required files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please ensure all RTL files are present"
    exit 1
fi

# Run Vivado project creation
echo ""
echo "=============================================="
echo "CREATING VIVADO PROJECT"
echo "=============================================="
echo "Command: vivado -mode batch -source create_vivado_project.tcl"
echo ""

# Set environment variables for better performance
export VIVADO_JOBS=4
export VIVADO_MEMORY_LIMIT=8192

# Run Vivado in batch mode
vivado -mode batch -source create_vivado_project.tcl

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✅ RTL BUILD COMPLETED SUCCESSFULLY!"
    echo "=============================================="
    echo ""
    echo "Generated files:"
    echo "  📁 Project: cnn_accelerator_rtl/"
    echo "  📊 Reports: cnn_accelerator_rtl/cnn_accelerator_rtl.runs/"
    echo "  📦 Bitstream: cnn_accelerator_rtl/cnn_accelerator_rtl.runs/impl_1/cnn_accelerator_rtl_wrapper.bit"
    echo ""
    echo "Next steps:"
    echo "1. Open Vivado IDE: vivado cnn_accelerator_rtl.xpr"
    echo "2. Review synthesis and implementation reports"
    echo "3. Program FPGA with bitstream"
    echo "4. Test CNN accelerator functionality"
    echo ""
    echo "To view reports:"
    echo "  ls -la cnn_accelerator_rtl/cnn_accelerator_rtl.runs/"
    echo ""
    echo "To program FPGA:"
    echo "  vivado -mode batch -source program_fpga.tcl"
else
    echo ""
    echo "=============================================="
    echo "❌ RTL BUILD FAILED!"
    echo "=============================================="
    echo "Check the error messages above for details."
    echo ""
    echo "Common issues:"
    echo "1. RTL syntax errors"
    echo "2. Constraint file issues"
    echo "3. Insufficient system resources"
    echo "4. Vivado toolchain issues"
    echo ""
    echo "To debug:"
    echo "1. Open Vivado IDE: vivado cnn_accelerator_rtl.xpr"
    echo "2. Check synthesis and implementation logs"
    echo "3. Review constraint file"
    exit 1
fi
