# CNN Accelerator RTL Implementation

## Overview

This directory contains the **direct RTL implementation** of the CNN accelerator for VR hand gesture recognition, targeting the Zybo Z7-20 FPGA. This approach bypasses HLS entirely and implements the CNN directly in Verilog RTL.

## 🎯 **Key Features**

- ✅ **Pure Vivado workflow** - No Vitis dependency
- ✅ **Direct RTL control** - Full control over implementation
- ✅ **INT4/INT8 quantization** - Optimized for FPGA resources
- ✅ **AXI interfaces** - Standard Xilinx IP interfaces
- ✅ **Memory optimization** - BRAM + DDR3 + Streaming strategy
- ✅ **319-layer CNN** - Complete hand gesture recognition model

## 📁 **File Structure**

```
fpga/rtl/
├── cnn_accelerator.v          # Main CNN accelerator module
├── conv_layer.v              # Convolution layer implementation
├── cnn_accelerator.xdc       # XDC constraints
├── create_vivado_project.tcl # Vivado project creation script
├── build_rtl.sh              # Build automation script
└── README_RTL.md            # This file
```

## 🏗️ **Architecture**

### **Main Components**

1. **CNN Accelerator (`cnn_accelerator.v`)**
   - AXI-Lite slave interface for control
   - AXI-MM master interface for DDR3 memory
   - State machine for layer processing
   - Feature map buffers (BRAM)
   - Weight buffer (DDR3)

2. **Convolution Layer (`conv_layer.v`)**
   - Configurable kernel size (3x3)
   - INT4 weights, INT8 activations
   - ReLU activation function
   - Parallel multiplier array
   - Accumulation logic

### **Memory Strategy**

- **BRAM**: Feature maps (1024 elements each)
- **DDR3**: Weight storage and large feature maps
- **Distributed RAM**: Weight buffers for active layers

### **Interface Design**

- **AXI-Lite**: Control registers and status
- **AXI-MM**: High-bandwidth memory access
- **Clock**: 200MHz from Zynq PS
- **Reset**: Active-low asynchronous reset

## 🚀 **Quick Start**

### **Prerequisites**

1. **Vivado 2025.1** installed and licensed
2. **Zybo Z7-20** board or compatible
3. **Linux environment** with bash shell

### **Build Steps**

1. **Navigate to RTL directory:**
   ```bash
   cd fpga/rtl
   ```

2. **Run the build script:**
   ```bash
   ./build_rtl.sh
   ```

3. **Monitor the build process:**
   - Synthesis (15-30 minutes)
   - Implementation (30-60 minutes)
   - Bitstream generation (5-10 minutes)

### **Expected Output**

```
=============================================
CNN ACCELERATOR RTL BUILD
=============================================
✅ RTL BUILD COMPLETED SUCCESSFULLY!

Generated files:
  📁 Project: cnn_accelerator_rtl/
  📊 Reports: cnn_accelerator_rtl/cnn_accelerator_rtl.runs/
  📦 Bitstream: cnn_accelerator_rtl/cnn_accelerator_rtl.runs/impl_1/cnn_accelerator_rtl_wrapper.bit
```

## 🔧 **Manual Build Process**

If you prefer to build manually:

1. **Source Vivado environment:**
   ```bash
   source /tools/Xilinx/2025.1/Vivado/.settings64-Vivado.sh
   ```

2. **Create project:**
   ```bash
   vivado -mode batch -source create_vivado_project.tcl
   ```

3. **Open in Vivado IDE:**
   ```bash
   vivado cnn_accelerator_rtl.xpr
   ```

## 📊 **Resource Utilization**

### **Target Specifications (Zybo Z7-20)**
- **FPGA**: XC7Z020CLG400-1
- **BRAM**: 280 blocks (max 70% = 196 blocks)
- **DSP**: 220 slices (max 80% = 176 slices)
- **LUT**: 53,200 (max 60% = 31,920)

### **Expected Utilization**
- **BRAM**: ~150 blocks (54%)
- **DSP**: ~160 slices (73%)
- **LUT**: ~25,000 (47%)
- **FF**: ~15,000 (28%)

## ⚡ **Performance Targets**

- **Clock Frequency**: 200MHz
- **Latency**: <1000 cycles (<5ms)
- **Throughput**: 30+ FPS
- **Power**: <2W (including Zynq PS)

## 🔍 **Debugging and Verification**

### **Simulation**

1. **Create testbench:**
   ```verilog
   // Add testbench files to project
   add_files -fileset sim_1 -norecurse testbench.v
   ```

2. **Run simulation:**
   ```bash
   launch_simulation
   run -all
   ```

### **Hardware Debug**

1. **Add debug cores:**
   ```tcl
   # In Vivado TCL console
   create_debug_core ila_0 ila
   set_property C_DATA_DEPTH 1024 [get_debug_cores ila_0]
   ```

2. **Program FPGA:**
   ```bash
   vivado -mode batch -source program_fpga.tcl
   ```

## 🐛 **Troubleshooting**

### **Common Issues**

1. **Synthesis fails:**
   - Check RTL syntax
   - Verify constraint file
   - Review resource utilization

2. **Timing violations:**
   - Check clock constraints
   - Review critical paths
   - Optimize pipeline stages

3. **Memory issues:**
   - Verify BRAM constraints
   - Check DDR3 interface
   - Review address mapping

### **Debug Commands**

```tcl
# Check synthesis status
get_property PROGRESS [get_runs synth_1]

# Check implementation status
get_property PROGRESS [get_runs impl_1]

# View timing report
open_run impl_1
report_timing_summary

# View utilization report
report_utilization
```

## 📈 **Optimization Strategies**

### **Performance Optimizations**

1. **Pipeline the convolution:**
   - Multiple pipeline stages
   - Parallel multiplier arrays
   - Optimized accumulation

2. **Memory access optimization:**
   - Burst transfers
   - Cache-friendly access patterns
   - Memory banking

3. **Resource sharing:**
   - Time-multiplexed processing
   - Shared arithmetic units
   - Efficient state machines

### **Area Optimizations**

1. **Resource constraints:**
   - Limit BRAM usage
   - Optimize DSP usage
   - Minimize LUT count

2. **Memory optimization:**
   - Efficient data packing
   - Compressed weight storage
   - Smart buffer management

## 🔄 **Integration with Software**

### **PS-PL Communication**

1. **AXI-Lite interface:**
   ```c
   // Control registers
   #define CTRL_REG     0x00
   #define INPUT_ADDR   0x04
   #define OUTPUT_ADDR  0x08
   #define WEIGHT_ADDR  0x0C
   #define STATUS_REG   0x10
   ```

2. **Memory mapping:**
   ```c
   // Map DDR3 memory
   void* input_buffer = mmap(NULL, INPUT_SIZE, PROT_READ|PROT_WRITE, 
                           MAP_SHARED, fd, DDR_BASE_ADDR);
   ```

### **Driver Integration**

1. **Linux driver:**
   - Character device driver
   - IOCTL interface
   - DMA support

2. **User application:**
   - OpenCV integration
   - Real-time processing
   - Result visualization

## 📚 **References**

- **Xilinx UG902**: Vivado Design Suite User Guide
- **Xilinx UG1037**: Vivado Design Suite TCL Command Reference
- **Zybo Z7-20 Reference Manual**: Digilent documentation
- **AXI Protocol Specification**: ARM AMBA documentation

## 🤝 **Contributing**

To contribute to this RTL implementation:

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Test thoroughly**
5. **Submit a pull request**

## 📄 **License**

This RTL implementation is part of the VR Hand Tracking Accelerator project and follows the same license terms.

---

**Engineer**: Sadad Haidari  
**Target**: Zybo Z7-20 (XC7Z020CLG400-1)  
**Architecture**: INT4/INT8 Quantized CNN  
**Performance**: 30+ FPS at 200MHz
