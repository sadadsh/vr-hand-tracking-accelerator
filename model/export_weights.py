"""
FPGA Weight Export & HLS Optimization
Optimizes quantized weights for HLS implementation and creates HLS-ready files

Engineer: Sadad Haidari
Input: weights.h + quantized model
Output: HLS-optimized weight files + memory layout + test data
Target: Zybo Z7-20 HLS CNN accelerator

source venv/bin/activate && python model/export_weights.py
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FPGAWeightExporter:
    """
    Export and optimize quantized weights for HLS CNN accelerator.
    Creates memory-efficient layouts and HLS-compatible data structures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.weights_dir = Path(config['weights_input_dir'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # HLS optimization parameters
        self.target_bram_kb = 630  # Zybo Z7-20 BRAM capacity
        self.max_parallel_ops = 16  # Target parallelization
        self.axi_width = 512  # AXI bus width for efficient transfers
        
        # Load quantization metadata
        self.metadata = self.load_quantization_metadata()
        self.layer_info = {}
        
        logger.info(f"FPGA Weight Exporter initialized")
        logger.info(f"Input directory: {self.weights_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_quantization_metadata(self) -> Dict[str, Any]:
        """Load quantization metadata from previous step."""
        metadata_file = self.weights_dir / 'quantization_metadata.json'
        
        if not metadata_file.exists():
            logger.warning(f"Metadata file not found: {metadata_file}")
            return {}
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"📊 Loaded quantization metadata:")
        print(f"   • Original accuracy: {metadata['model_info']['original_accuracy']:.2f}%")
        print(f"   • Quantized accuracy: {metadata['model_info']['quantized_accuracy']:.2f}%")
        print(f"   • Compression ratio: {metadata['model_info']['compression_ratio']:.1f}x")
        print(f"   • Model size: {metadata['model_info']['quantized_size_kb']:.1f} KB")
        
        return metadata
    
    def parse_weights_header(self) -> Dict[str, Dict[str, Any]]:
        """Parse the generated weights.h file to extract layer information."""
        print("\n🔍 Parsing weights.h file...")
        
        weights_file = self.weights_dir / 'weights.h'
        if not weights_file.exists():
            raise FileNotFoundError(f"weights.h not found: {weights_file}")
        
        layer_info = {}
        
        with open(weights_file, 'r') as f:
            content = f.read()
        
        # Parse layer information using regex
        layer_pattern = r'// Layer: (.+?)\n// Original shape: \[(.+?)\]\n// Scale factor: (.+?)\n// Packed INT4 values: (\d+)\nconst ap_uint<8> (.+?)_weights\[(\d+)\]'
        
        matches = re.findall(layer_pattern, content)
        
        for match in matches:
            layer_name = match[0]
            shape_str = match[1]
            scale_factor = float(match[2])
            packed_values = int(match[3])
            array_name = match[4]
            array_size = int(match[5])
            
            # Parse shape
            shape = [int(x.strip()) for x in shape_str.split(',')]
            
            layer_info[layer_name] = {
                'shape': shape,
                'scale_factor': scale_factor,
                'packed_values': packed_values,
                'array_name': array_name,
                'array_size': array_size,
                'memory_kb': array_size / 1024
            }
            
            print(f"   • {layer_name}: {shape} → {array_size} bytes ({array_size/1024:.1f} KB)")
        
        total_memory_kb = sum(info['memory_kb'] for info in layer_info.values())
        print(f"✅ Parsed {len(layer_info)} layers, total: {total_memory_kb:.1f} KB")
        
        self.layer_info = layer_info
        return layer_info
    
    def analyze_memory_layout(self) -> Dict[str, Any]:
        """Analyze memory requirements and create optimal layout strategy."""
        print("\n🧠 Analyzing memory layout for FPGA...")
        
        total_memory_kb = sum(info['memory_kb'] for info in self.layer_info.values())
        
        # Categorize layers by memory requirements
        small_layers = []  # <10KB - can fit in BRAM
        medium_layers = []  # 10-100KB - may need DDR3
        large_layers = []  # >100KB - definitely need DDR3
        
        for layer_name, info in self.layer_info.items():
            size_kb = info['memory_kb']
            if size_kb < 10:
                small_layers.append((layer_name, info))
            elif size_kb < 100:
                medium_layers.append((layer_name, info))
            else:
                large_layers.append((layer_name, info))
        
        # Determine memory allocation strategy
        bram_allocated = 0
        memory_strategy = {
            'bram_layers': [],
            'ddr_layers': [],
            'streaming_layers': []
        }
        
        # Priority: Put small, frequently accessed layers in BRAM
        for layer_name, info in sorted(small_layers, key=lambda x: x[1]['memory_kb']):
            if bram_allocated + info['memory_kb'] <= self.target_bram_kb * 0.8:  # 80% utilization
                memory_strategy['bram_layers'].append(layer_name)
                bram_allocated += info['memory_kb']
            else:
                memory_strategy['ddr_layers'].append(layer_name)
        
        # Medium layers go to DDR3 or streaming
        for layer_name, info in medium_layers:
            memory_strategy['ddr_layers'].append(layer_name)
        
        # Large layers use streaming approach
        for layer_name, info in large_layers:
            memory_strategy['streaming_layers'].append(layer_name)
        
        analysis = {
            'total_memory_kb': total_memory_kb,
            'bram_utilization_kb': bram_allocated,
            'bram_utilization_percent': (bram_allocated / self.target_bram_kb) * 100,
            'num_small_layers': len(small_layers),
            'num_medium_layers': len(medium_layers),
            'num_large_layers': len(large_layers),
            'memory_strategy': memory_strategy,
            'fits_in_bram': total_memory_kb <= self.target_bram_kb
        }
        
        print(f"📋 Memory Analysis:")
        print(f"   • Total model size: {total_memory_kb:.1f} KB")
        print(f"   • BRAM capacity: {self.target_bram_kb} KB")
        print(f"   • BRAM utilization: {bram_allocated:.1f} KB ({analysis['bram_utilization_percent']:.1f}%)")
        print(f"   • Small layers (<10KB): {len(small_layers)}")
        print(f"   • Medium layers (10-100KB): {len(medium_layers)}")
        print(f"   • Large layers (>100KB): {len(large_layers)}")
        
        if analysis['fits_in_bram']:
            print(f"   ✅ Entire model fits in BRAM!")
        else:
            print(f"   📤 Hybrid BRAM + DDR3 strategy needed")
        
        return analysis
    
    def generate_hls_layer_headers(self, memory_analysis: Dict[str, Any]):
        """Generate optimized HLS header files for each layer type."""
        print("\n🏭 Generating HLS layer headers...")
        
        hls_dir = self.output_dir / 'hls_headers'
        hls_dir.mkdir(exist_ok=True)
        
        # Generate BRAM layers header
        self._generate_bram_layers_header(memory_analysis, hls_dir)
        
        # Generate DDR3 layers header
        self._generate_ddr_layers_header(memory_analysis, hls_dir)
        
        # Generate streaming layers header
        self._generate_streaming_layers_header(memory_analysis, hls_dir)
        
        # Generate main layer configuration
        self._generate_layer_config_header(memory_analysis, hls_dir)
        
        print(f"✅ HLS headers generated in: {hls_dir}")
    
    def _generate_bram_layers_header(self, memory_analysis: Dict[str, Any], hls_dir: Path):
        """Generate header for BRAM-stored layers."""
        bram_file = hls_dir / 'bram_layers.h'
        
        with open(bram_file, 'w') as f:
            f.write("/*\n")
            f.write(" * BRAM-Stored CNN Layers\n")
            f.write(" * Fast access layers stored in FPGA BRAM\n")
            f.write(" * Generated for HLS CNN accelerator\n")
            f.write(" */\n\n")
            f.write("#ifndef BRAM_LAYERS_H\n")
            f.write("#define BRAM_LAYERS_H\n\n")
            f.write("#include <ap_int.h>\n")
            f.write("#include <ap_fixed.h>\n\n")
            
            bram_layers = memory_analysis['memory_strategy']['bram_layers']
            total_bram_size = 0
            
            for layer_name in bram_layers:
                info = self.layer_info[layer_name]
                clean_name = layer_name.replace('.', '_').replace('/', '_')
                
                f.write(f"// BRAM Layer: {layer_name}\n")
                f.write(f"// Shape: {info['shape']}\n")
                f.write(f"// Memory: {info['memory_kb']:.1f} KB\n")
                f.write(f"#define {clean_name.upper()}_SIZE {info['array_size']}\n")
                f.write(f"extern const ap_uint<8> {clean_name}_weights[{info['array_size']}];\n")
                f.write(f"extern const float {clean_name}_scale;\n\n")
                
                total_bram_size += info['memory_kb']
            
            f.write(f"// BRAM Summary\n")
            f.write(f"#define NUM_BRAM_LAYERS {len(bram_layers)}\n")
            f.write(f"#define TOTAL_BRAM_SIZE_KB {total_bram_size:.1f}\n")
            f.write(f"#define BRAM_UTILIZATION_PERCENT {(total_bram_size/self.target_bram_kb)*100:.1f}\n\n")
            f.write("#endif // BRAM_LAYERS_H\n")
        
        print(f"   • BRAM layers: {len(bram_layers)} layers, {total_bram_size:.1f} KB")
    
    def _generate_ddr_layers_header(self, memory_analysis: Dict[str, Any], hls_dir: Path):
        """Generate header for DDR3-stored layers."""
        ddr_file = hls_dir / 'ddr_layers.h'
        
        with open(ddr_file, 'w') as f:
            f.write("/*\n")
            f.write(" * DDR3-Stored CNN Layers\n")
            f.write(" * Large layers stored in external DDR3 memory\n")
            f.write(" * Generated for HLS CNN accelerator\n")
            f.write(" */\n\n")
            f.write("#ifndef DDR_LAYERS_H\n")
            f.write("#define DDR_LAYERS_H\n\n")
            f.write("#include <ap_int.h>\n")
            f.write("#include <ap_fixed.h>\n\n")
            
            ddr_layers = memory_analysis['memory_strategy']['ddr_layers']
            total_ddr_size = 0
            
            # Generate AXI interface for DDR3 access
            f.write("// AXI Interface for DDR3 Access\n")
            f.write("typedef struct {\n")
            f.write("    ap_uint<32> base_addr;\n")
            f.write("    ap_uint<32> size_bytes;\n")
            f.write("    float scale_factor;\n")
            f.write("    ap_uint<16> shape[4];  // [N, C, H, W]\n")
            f.write("} ddr_layer_info_t;\n\n")
            
            f.write("// DDR3 Layer Definitions\n")
            for i, layer_name in enumerate(ddr_layers):
                info = self.layer_info[layer_name]
                clean_name = layer_name.replace('.', '_').replace('/', '_')
                
                f.write(f"// DDR Layer {i}: {layer_name}\n")
                f.write(f"// Shape: {info['shape']}\n")
                f.write(f"// Memory: {info['memory_kb']:.1f} KB\n")
                f.write(f"#define {clean_name.upper()}_DDR_OFFSET 0x{int(total_ddr_size*1024):08X}\n")
                f.write(f"#define {clean_name.upper()}_DDR_SIZE {info['array_size']}\n\n")
                
                total_ddr_size += info['memory_kb']
            
            f.write(f"// DDR3 Summary\n")
            f.write(f"#define NUM_DDR_LAYERS {len(ddr_layers)}\n")
            f.write(f"#define TOTAL_DDR_SIZE_KB {total_ddr_size:.1f}\n")
            f.write(f"#define DDR_BASE_ADDRESS 0x00000000\n\n")
            
            # Generate layer info array
            f.write("const ddr_layer_info_t ddr_layer_info[NUM_DDR_LAYERS] = {\n")
            offset = 0
            for i, layer_name in enumerate(ddr_layers):
                info = self.layer_info[layer_name]
                shape_padded = info['shape'] + [1] * (4 - len(info['shape']))  # Pad to 4D
                
                f.write(f"    {{0x{offset:08X}, {info['array_size']}, {info['scale_factor']:.6f}f, ")
                f.write(f"{{{', '.join(map(str, shape_padded[:4]))}}}}}")
                if i < len(ddr_layers) - 1:
                    f.write(",")
                f.write(f"  // {layer_name}\n")
                
                offset += info['array_size']
            
            f.write("};\n\n")
            f.write("#endif // DDR_LAYERS_H\n")
        
        print(f"   • DDR3 layers: {len(ddr_layers)} layers, {total_ddr_size:.1f} KB")
    
    def _generate_streaming_layers_header(self, memory_analysis: Dict[str, Any], hls_dir: Path):
        """Generate header for streaming layers (very large layers)."""
        streaming_file = hls_dir / 'streaming_layers.h'
        
        with open(streaming_file, 'w') as f:
            f.write("/*\n")
            f.write(" * Streaming CNN Layers\n")
            f.write(" * Very large layers processed with weight streaming\n")
            f.write(" * Generated for HLS CNN accelerator\n")
            f.write(" */\n\n")
            f.write("#ifndef STREAMING_LAYERS_H\n")
            f.write("#define STREAMING_LAYERS_H\n\n")
            f.write("#include <ap_int.h>\n")
            f.write("#include <hls_stream.h>\n\n")
            
            streaming_layers = memory_analysis['memory_strategy']['streaming_layers']
            
            if streaming_layers:
                f.write("// Streaming Configuration\n")
                f.write(f"#define STREAM_BUFFER_SIZE {self.axi_width // 8}\n")  # Bytes per transfer
                f.write(f"#define MAX_STREAM_CHUNKS 1024\n\n")
                
                f.write("typedef struct {\n")
                f.write("    ap_uint<32> total_size;\n")
                f.write("    ap_uint<16> chunk_size;\n")
                f.write("    ap_uint<16> num_chunks;\n")
                f.write("    float scale_factor;\n")
                f.write("} stream_layer_info_t;\n\n")
                
                total_streaming_size = 0
                for i, layer_name in enumerate(streaming_layers):
                    info = self.layer_info[layer_name]
                    clean_name = layer_name.replace('.', '_').replace('/', '_')
                    
                    f.write(f"// Streaming Layer {i}: {layer_name}\n")
                    f.write(f"// Shape: {info['shape']}\n")
                    f.write(f"// Memory: {info['memory_kb']:.1f} KB\n")
                    f.write(f"#define {clean_name.upper()}_CHUNKS {info['array_size'] // (self.axi_width // 8) + 1}\n\n")
                    
                    total_streaming_size += info['memory_kb']
                
                f.write(f"#define NUM_STREAMING_LAYERS {len(streaming_layers)}\n")
                f.write(f"#define TOTAL_STREAMING_SIZE_KB {total_streaming_size:.1f}\n")
            else:
                f.write("// No streaming layers needed\n")
                f.write("#define NUM_STREAMING_LAYERS 0\n")
            
            f.write("\n#endif // STREAMING_LAYERS_H\n")
        
        print(f"   • Streaming layers: {len(streaming_layers)} layers")
    
    def _generate_layer_config_header(self, memory_analysis: Dict[str, Any], hls_dir: Path):
        """Generate main layer configuration header."""
        config_file = hls_dir / 'layer_config.h'
        
        with open(config_file, 'w') as f:
            f.write("/*\n")
            f.write(" * CNN Layer Configuration\n")
            f.write(" * Main configuration file for HLS CNN accelerator\n")
            f.write(" * Generated for Zybo Z7-20 FPGA\n")
            f.write(" */\n\n")
            f.write("#ifndef LAYER_CONFIG_H\n")
            f.write("#define LAYER_CONFIG_H\n\n")
            f.write("#include \"bram_layers.h\"\n")
            f.write("#include \"ddr_layers.h\"\n")
            f.write("#include \"streaming_layers.h\"\n\n")
            
            # Overall configuration
            f.write("// Overall Model Configuration\n")
            f.write(f"#define TOTAL_LAYERS {len(self.layer_info)}\n")
            f.write(f"#define NUM_CLASSES {self.config.get('num_classes', 18)}\n")
            f.write(f"#define INPUT_SIZE {self.config.get('input_size', 224)}\n")
            f.write(f"#define MODEL_MEMORY_KB {memory_analysis['total_memory_kb']:.1f}\n\n")
            
            # Memory strategy summary
            f.write("// Memory Strategy\n")
            bram_layers = memory_analysis['memory_strategy']['bram_layers']
            ddr_layers = memory_analysis['memory_strategy']['ddr_layers']
            streaming_layers = memory_analysis['memory_strategy']['streaming_layers']
            
            f.write(f"#define STRATEGY_BRAM_LAYERS {len(bram_layers)}\n")
            f.write(f"#define STRATEGY_DDR_LAYERS {len(ddr_layers)}\n")
            f.write(f"#define STRATEGY_STREAMING_LAYERS {len(streaming_layers)}\n\n")
            
            # Performance targets
            f.write("// Performance Targets\n")
            f.write("#define TARGET_CLOCK_MHZ 200\n")
            f.write("#define TARGET_LATENCY_CYCLES 1000\n")
            f.write("#define MAX_PARALLEL_OPS 16\n")
            f.write("#define AXI_DATA_WIDTH 512\n\n")
            
            # Layer type enumeration
            f.write("typedef enum {\n")
            f.write("    LAYER_TYPE_CONV2D,\n")
            f.write("    LAYER_TYPE_RELU,\n")
            f.write("    LAYER_TYPE_POOL,\n")
            f.write("    LAYER_TYPE_FC,\n")
            f.write("    LAYER_TYPE_CLASSIFIER\n")
            f.write("} layer_type_t;\n\n")
            
            f.write("typedef enum {\n")
            f.write("    MEMORY_TYPE_BRAM,\n")
            f.write("    MEMORY_TYPE_DDR,\n")
            f.write("    MEMORY_TYPE_STREAM\n")
            f.write("} memory_type_t;\n\n")
            
            f.write("#endif // LAYER_CONFIG_H\n")
        
        print(f"   • Main config: {config_file}")
    
    def generate_hls_testbench(self):
        """Generate HLS testbench with sample input data."""
        print("\n🧪 Generating HLS testbench...")
        
        tb_dir = self.output_dir / 'testbench'
        tb_dir.mkdir(exist_ok=True)
        
        # Generate test input data
        test_input_file = tb_dir / 'test_input.h'
        
        with open(test_input_file, 'w') as f:
            f.write("/*\n")
            f.write(" * Test Input Data for HLS CNN Accelerator\n")
            f.write(" * Sample 224x224x3 image for verification\n")
            f.write(" */\n\n")
            f.write("#ifndef TEST_INPUT_H\n")
            f.write("#define TEST_INPUT_H\n\n")
            f.write("#include <ap_int.h>\n\n")
            
            # Generate sample normalized input (224x224x3)
            input_size = self.config.get('input_size', 224)
            total_pixels = input_size * input_size * 3
            
            f.write(f"#define INPUT_HEIGHT {input_size}\n")
            f.write(f"#define INPUT_WIDTH {input_size}\n")
            f.write(f"#define INPUT_CHANNELS 3\n")
            f.write(f"#define TOTAL_INPUT_SIZE {total_pixels}\n\n")
            
            # Generate sample input (simple gradient pattern)
            f.write(f"// Sample input data (normalized to 0-255 range)\n")
            f.write(f"const ap_uint<8> test_input_data[TOTAL_INPUT_SIZE] = {{\n")
            
            # Create a simple test pattern
            for h in range(input_size):
                if h % 16 == 0:
                    f.write("    ")
                for w in range(input_size):
                    for c in range(3):
                        # Simple gradient pattern
                        value = (h + w + c * 50) % 256
                        f.write(f"{value}")
                        if h < input_size - 1 or w < input_size - 1 or c < 2:
                            f.write(", ")
                        
                        if (h * input_size * 3 + w * 3 + c + 1) % 16 == 0:
                            f.write("\n    ")
            
            f.write("\n};\n\n")
            
            # Expected output (placeholder)
            f.write("// Expected output (18 classes)\n")
            f.write("const float expected_output[18] = {\n")
            f.write("    0.1, 0.05, 0.15, 0.08, 0.12, 0.06, 0.2, 0.04,\n")
            f.write("    0.03, 0.07, 0.09, 0.11, 0.13, 0.02, 0.01, 0.05,\n")
            f.write("    0.08, 0.12\n")
            f.write("};\n\n")
            f.write("#endif // TEST_INPUT_H\n")
        
        # Generate main testbench
        testbench_file = tb_dir / 'cnn_testbench.cpp'
        
        with open(testbench_file, 'w') as f:
            f.write("/*\n")
            f.write(" * HLS CNN Accelerator Testbench\n")
            f.write(" * Verifies CNN inference with sample input\n")
            f.write(" */\n\n")
            f.write("#include <iostream>\n")
            f.write("#include <cmath>\n")
            f.write("#include \"../hls_headers/layer_config.h\"\n")
            f.write("#include \"test_input.h\"\n\n")
            f.write("// Forward declaration of CNN accelerator\n")
            f.write("void cnn_accelerator(\n")
            f.write("    ap_uint<8> input_data[TOTAL_INPUT_SIZE],\n")
            f.write("    float output_data[NUM_CLASSES]\n")
            f.write(");\n\n")
            f.write("int main() {\n")
            f.write("    std::cout << \"Starting CNN Accelerator Testbench...\\n\";\n\n")
            f.write("    // Prepare input data\n")
            f.write("    ap_uint<8> input_buffer[TOTAL_INPUT_SIZE];\n")
            f.write("    float output_buffer[NUM_CLASSES];\n\n")
            f.write("    // Copy test input\n")
            f.write("    for (int i = 0; i < TOTAL_INPUT_SIZE; i++) {\n")
            f.write("        input_buffer[i] = test_input_data[i];\n")
            f.write("    }\n\n")
            f.write("    std::cout << \"Running CNN inference...\\n\";\n")
            f.write("    cnn_accelerator(input_buffer, output_buffer);\n\n")
            f.write("    // Print results\n")
            f.write("    std::cout << \"Output probabilities:\\n\";\n")
            f.write("    for (int i = 0; i < NUM_CLASSES; i++) {\n")
            f.write("        std::cout << \"Class \" << i << \": \" << output_buffer[i] << \"\\n\";\n")
            f.write("    }\n\n")
            f.write("    // Find max prediction\n")
            f.write("    int max_class = 0;\n")
            f.write("    float max_prob = output_buffer[0];\n")
            f.write("    for (int i = 1; i < NUM_CLASSES; i++) {\n")
            f.write("        if (output_buffer[i] > max_prob) {\n")
            f.write("            max_prob = output_buffer[i];\n")
            f.write("            max_class = i;\n")
            f.write("        }\n")
            f.write("    }\n\n")
            f.write("    std::cout << \"Predicted class: \" << max_class << \" (confidence: \" << max_prob << \")\\n\";\n")
            f.write("    std::cout << \"Testbench completed successfully!\\n\";\n\n")
            f.write("    return 0;\n")
            f.write("}\n")
        
        print(f"✅ Testbench generated:")
        print(f"   • Test input: {test_input_file}")
        print(f"   • Testbench: {testbench_file}")
    
    def generate_build_scripts(self):
        """Generate build scripts for HLS project."""
        print("\n🔨 Generating build scripts...")
        
        scripts_dir = self.output_dir / 'build_scripts'
        scripts_dir.mkdir(exist_ok=True)
        
        # Generate HLS build TCL script
        hls_tcl = scripts_dir / 'build_hls.tcl'
        
        with open(hls_tcl, 'w') as f:
            f.write("# HLS CNN Accelerator Build Script\n")
            f.write("# Generated for Zybo Z7-20 FPGA\n\n")
            f.write("# Create new project\n")
            f.write("open_project cnn_accelerator_hls\n")
            f.write("set_top cnn_accelerator\n\n")
            f.write("# Add source files\n")
            f.write("add_files ../hls_src/cnn_accelerator.cpp\n")
            f.write("add_files ../hls_headers/layer_config.h\n")
            f.write("add_files ../hls_headers/bram_layers.h\n")
            f.write("add_files ../hls_headers/ddr_layers.h\n")
            f.write("add_files ../hls_headers/streaming_layers.h\n")
            f.write("add_files ../weights.h\n\n")
            f.write("# Add testbench\n")
            f.write("add_files -tb ../testbench/cnn_testbench.cpp\n")
            f.write("add_files -tb ../testbench/test_input.h\n\n")
            f.write("# Create solution\n")
            f.write("open_solution \"solution1\"\n")
            f.write("set_part {xc7z020clg400-1}  # Zybo Z7-20\n")
            f.write("create_clock -period 5 -name default  # 200MHz\n\n")
            f.write("# Synthesis\n")
            f.write("csynth_design\n\n")
            f.write("# Co-simulation (optional)\n")
            f.write("# cosim_design\n\n")
            f.write("# Export IP\n")
            f.write("export_design -format ip_catalog\n\n")
            f.write("exit\n")
        
        # Generate Makefile
        makefile = scripts_dir / 'Makefile'
        
        with open(makefile, 'w') as f:
            f.write("# Makefile for CNN Accelerator HLS Project\n\n")
            f.write("PROJECT = cnn_accelerator_hls\n")
            f.write("SOLUTION = solution1\n")
            f.write("TOP_FUNCTION = cnn_accelerator\n\n")
            f.write(".PHONY: all clean synthesis cosim export\n\n")
            f.write("all: synthesis\n\n")
            f.write("synthesis:\n")
            f.write("\t@echo \"Running HLS synthesis...\"\n")
            f.write("\tvivado_hls -f build_hls.tcl\n\n")
            f.write("cosim: synthesis\n")
            f.write("\t@echo \"Running co-simulation...\"\n")
            f.write("\t# Add co-simulation commands here\n\n")
            f.write("export: synthesis\n")
            f.write("\t@echo \"Exporting IP...\"\n")
            f.write("\t# IP export is included in build_hls.tcl\n\n")
            f.write("clean:\n")
            f.write("\t@echo \"Cleaning project...\"\n")
            f.write("\trm -rf $(PROJECT)\n")
            f.write("\trm -rf *.log\n\n")
            f.write("help:\n")
            f.write("\t@echo \"Available targets:\"\n")
            f.write("\t@echo \"  synthesis - Run HLS synthesis\"\n")
            f.write("\t@echo \"  cosim     - Run co-simulation\"\n")
            f.write("\t@echo \"  export    - Export IP\"\n")
            f.write("\t@echo \"  clean     - Clean project\"\n")
        
        print(f"✅ Build scripts generated:")
        print(f"   • HLS TCL: {hls_tcl}")
        print(f"   • Makefile: {makefile}")
    
    def generate_summary_report(self, memory_analysis: Dict[str, Any]):
        """Generate comprehensive summary report."""
        print("\n📊 Generating summary report...")
        
        report_file = self.output_dir / 'fpga_export_summary.json'
        
        summary = {
            'model_info': {
                'total_layers': len(self.layer_info),
                'total_memory_kb': memory_analysis['total_memory_kb'],
                'quantized_accuracy': self.metadata.get('model_info', {}).get('quantized_accuracy', 'Unknown'),
                'compression_ratio': self.metadata.get('model_info', {}).get('compression_ratio', 'Unknown')
            },
            'fpga_deployment': {
                'target_device': 'Zybo Z7-20',
                'bram_capacity_kb': self.target_bram_kb,
                'bram_utilization_kb': memory_analysis['bram_utilization_kb'],
                'bram_utilization_percent': memory_analysis['bram_utilization_percent'],
                'fits_in_bram': memory_analysis['fits_in_bram']
            },
            'memory_strategy': memory_analysis['memory_strategy'],
            'performance_targets': {
                'clock_frequency_mhz': 200,
                'target_latency_ms': 1,
                'max_parallel_ops': self.max_parallel_ops,
                'axi_data_width': self.axi_width
            },
            'generated_files': {
                'hls_headers': 'hls_headers/',
                'testbench': 'testbench/',
                'build_scripts': 'build_scripts/',
                'summary': 'fpga_export_summary.json'
            },
            'next_steps': [
                'Implement CNN accelerator in HLS (cnn_accelerator.cpp)',
                'Run HLS synthesis using build_hls.tcl',
                'Integrate IP into Vivado project',
                'Test with hardware-in-the-loop'
            ]
        }
        
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Summary report saved: {report_file}")
        return summary
    
    def export_weights(self) -> bool:
        """Main weight export pipeline."""
        try:
            print("\n" + "="*80)
            print("FPGA WEIGHT EXPORT & HLS OPTIMIZATION")
            print("="*80)
            print(f"Target: Zybo Z7-20 FPGA with 630KB BRAM")
            print(f"Goal: HLS-optimized weight layout and test infrastructure")
            
            # Step 1: Parse weights header
            layer_info = self.parse_weights_header()
            
            # Step 2: Analyze memory layout
            memory_analysis = self.analyze_memory_layout()
            
            # Step 3: Generate HLS headers
            self.generate_hls_layer_headers(memory_analysis)
            
            # Step 4: Generate testbench
            self.generate_hls_testbench()
            
            # Step 5: Generate build scripts
            self.generate_build_scripts()
            
            # Step 6: Generate summary
            self.generate_summary_report(memory_analysis)
            
            print(f"\n" + "="*80)
            print("FPGA EXPORT COMPLETED!")
            print("="*80)
            print(f"✅ Total layers: {len(layer_info)}")
            print(f"✅ Model size: {memory_analysis['total_memory_kb']:.1f} KB")
            print(f"✅ BRAM utilization: {memory_analysis['bram_utilization_percent']:.1f}%")
            print(f"✅ Memory strategy: {'BRAM-only' if memory_analysis['fits_in_bram'] else 'Hybrid BRAM+DDR3'}")
            print(f"✅ Files generated in: {self.output_dir}")
            print(f"✅ Ready for HLS implementation!")
            
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description='Export and optimize weights for HLS implementation')
    parser.add_argument('--weights_dir', '-w', type=str,
                       default='model/quantized',
                       help='Directory containing weights.h and metadata')
    parser.add_argument('--output_dir', '-o', type=str,
                       default='fpga/hls_weights',
                       help='Output directory for HLS-optimized files')
    parser.add_argument('--num_classes', '-c', type=int, default=18,
                       help='Number of output classes')
    parser.add_argument('--input_size', '-s', type=int, default=224,
                       help='Input image size')
    
    args = parser.parse_args()
    
    config = {
        'weights_input_dir': args.weights_dir,
        'output_dir': args.output_dir,
        'num_classes': args.num_classes,
        'input_size': args.input_size
    }
    
    print("FPGA Weight Export & HLS Optimization")
    print("="*50)
    print(f"Input directory: {config['weights_input_dir']}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Classes: {config['num_classes']}")
    print(f"Input size: {config['input_size']}x{config['input_size']}")
    print(f"Target: Zybo Z7-20 (630KB BRAM)")
    
    # Confirm export
    response = input("\nStart HLS optimization? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Export cancelled.")
        return
    
    # Create exporter and start
    exporter = FPGAWeightExporter(config)
    success = exporter.export_weights()
    
    if success:
        print(f"\n🎉 SUCCESS! Weights optimized for HLS implementation")
        print(f"📁 Files generated in: {config['output_dir']}")
        print(f"🔧 Next step: Implement CNN accelerator in HLS")
    else:
        print(f"\n❌ Export failed! Check the logs above.")


if __name__ == "__main__":
    main()