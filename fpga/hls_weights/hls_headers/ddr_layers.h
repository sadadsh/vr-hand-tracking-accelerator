/*
 * DDR3-Stored CNN Layers
 * Large layers stored in external DDR3 memory
 * Generated for HLS CNN accelerator
 */

#ifndef DDR_LAYERS_H
#define DDR_LAYERS_H

#include <ap_int.h>
#include <ap_fixed.h>

// AXI Interface for DDR3 Access
typedef struct {
    ap_uint<32> base_addr;
    ap_uint<32> size_bytes;
    float scale_factor;
    ap_uint<16> shape[4];  // [N, C, H, W]
} ddr_layer_info_t;

// DDR3 Layer Definitions
// DDR Layer 0: backbone.layer1.0.conv2
// Shape: [64, 64, 3, 3]
// Memory: 18.0 KB
#define BACKBONE_LAYER1_0_CONV2_DDR_OFFSET 0x00000000
#define BACKBONE_LAYER1_0_CONV2_DDR_SIZE 18432

// DDR Layer 1: backbone.layer1.1.conv2
// Shape: [64, 64, 3, 3]
// Memory: 18.0 KB
#define BACKBONE_LAYER1_1_CONV2_DDR_OFFSET 0x00004800
#define BACKBONE_LAYER1_1_CONV2_DDR_SIZE 18432

// DDR Layer 2: backbone.layer1.2.conv2
// Shape: [64, 64, 3, 3]
// Memory: 18.0 KB
#define BACKBONE_LAYER1_2_CONV2_DDR_OFFSET 0x00009000
#define BACKBONE_LAYER1_2_CONV2_DDR_SIZE 18432

// DDR Layer 3: backbone.layer2.0.conv1
// Shape: [128, 256, 1, 1]
// Memory: 16.0 KB
#define BACKBONE_LAYER2_0_CONV1_DDR_OFFSET 0x0000D800
#define BACKBONE_LAYER2_0_CONV1_DDR_SIZE 16384

// DDR Layer 4: backbone.layer2.0.conv2
// Shape: [128, 128, 3, 3]
// Memory: 72.0 KB
#define BACKBONE_LAYER2_0_CONV2_DDR_OFFSET 0x00011800
#define BACKBONE_LAYER2_0_CONV2_DDR_SIZE 73728

// DDR Layer 5: backbone.layer2.0.conv3
// Shape: [512, 128, 1, 1]
// Memory: 32.0 KB
#define BACKBONE_LAYER2_0_CONV3_DDR_OFFSET 0x00023800
#define BACKBONE_LAYER2_0_CONV3_DDR_SIZE 32768

// DDR Layer 6: backbone.layer2.0.downsample.0
// Shape: [512, 256, 1, 1]
// Memory: 64.0 KB
#define BACKBONE_LAYER2_0_DOWNSAMPLE_0_DDR_OFFSET 0x0002B800
#define BACKBONE_LAYER2_0_DOWNSAMPLE_0_DDR_SIZE 65536

// DDR Layer 7: backbone.layer2.1.conv1
// Shape: [128, 512, 1, 1]
// Memory: 32.0 KB
#define BACKBONE_LAYER2_1_CONV1_DDR_OFFSET 0x0003B800
#define BACKBONE_LAYER2_1_CONV1_DDR_SIZE 32768

// DDR Layer 8: backbone.layer2.1.conv2
// Shape: [128, 128, 3, 3]
// Memory: 72.0 KB
#define BACKBONE_LAYER2_1_CONV2_DDR_OFFSET 0x00043800
#define BACKBONE_LAYER2_1_CONV2_DDR_SIZE 73728

// DDR Layer 9: backbone.layer2.1.conv3
// Shape: [512, 128, 1, 1]
// Memory: 32.0 KB
#define BACKBONE_LAYER2_1_CONV3_DDR_OFFSET 0x00055800
#define BACKBONE_LAYER2_1_CONV3_DDR_SIZE 32768

// DDR Layer 10: backbone.layer2.2.conv1
// Shape: [128, 512, 1, 1]
// Memory: 32.0 KB
#define BACKBONE_LAYER2_2_CONV1_DDR_OFFSET 0x0005D800
#define BACKBONE_LAYER2_2_CONV1_DDR_SIZE 32768

// DDR Layer 11: backbone.layer2.2.conv2
// Shape: [128, 128, 3, 3]
// Memory: 72.0 KB
#define BACKBONE_LAYER2_2_CONV2_DDR_OFFSET 0x00065800
#define BACKBONE_LAYER2_2_CONV2_DDR_SIZE 73728

// DDR Layer 12: backbone.layer2.2.conv3
// Shape: [512, 128, 1, 1]
// Memory: 32.0 KB
#define BACKBONE_LAYER2_2_CONV3_DDR_OFFSET 0x00077800
#define BACKBONE_LAYER2_2_CONV3_DDR_SIZE 32768

// DDR Layer 13: backbone.layer2.3.conv1
// Shape: [128, 512, 1, 1]
// Memory: 32.0 KB
#define BACKBONE_LAYER2_3_CONV1_DDR_OFFSET 0x0007F800
#define BACKBONE_LAYER2_3_CONV1_DDR_SIZE 32768

// DDR Layer 14: backbone.layer2.3.conv2
// Shape: [128, 128, 3, 3]
// Memory: 72.0 KB
#define BACKBONE_LAYER2_3_CONV2_DDR_OFFSET 0x00087800
#define BACKBONE_LAYER2_3_CONV2_DDR_SIZE 73728

// DDR Layer 15: backbone.layer2.3.conv3
// Shape: [512, 128, 1, 1]
// Memory: 32.0 KB
#define BACKBONE_LAYER2_3_CONV3_DDR_OFFSET 0x00099800
#define BACKBONE_LAYER2_3_CONV3_DDR_SIZE 32768

// DDR Layer 16: backbone.layer2.4.conv1
// Shape: [128, 512, 1, 1]
// Memory: 32.0 KB
#define BACKBONE_LAYER2_4_CONV1_DDR_OFFSET 0x000A1800
#define BACKBONE_LAYER2_4_CONV1_DDR_SIZE 32768

// DDR Layer 17: backbone.layer2.4.conv2
// Shape: [128, 128, 3, 3]
// Memory: 72.0 KB
#define BACKBONE_LAYER2_4_CONV2_DDR_OFFSET 0x000A9800
#define BACKBONE_LAYER2_4_CONV2_DDR_SIZE 73728

// DDR Layer 18: backbone.layer2.4.conv3
// Shape: [512, 128, 1, 1]
// Memory: 32.0 KB
#define BACKBONE_LAYER2_4_CONV3_DDR_OFFSET 0x000BB800
#define BACKBONE_LAYER2_4_CONV3_DDR_SIZE 32768

// DDR Layer 19: backbone.layer2.5.conv1
// Shape: [128, 512, 1, 1]
// Memory: 32.0 KB
#define BACKBONE_LAYER2_5_CONV1_DDR_OFFSET 0x000C3800
#define BACKBONE_LAYER2_5_CONV1_DDR_SIZE 32768

// DDR Layer 20: backbone.layer2.5.conv2
// Shape: [128, 128, 3, 3]
// Memory: 72.0 KB
#define BACKBONE_LAYER2_5_CONV2_DDR_OFFSET 0x000CB800
#define BACKBONE_LAYER2_5_CONV2_DDR_SIZE 73728

// DDR Layer 21: backbone.layer2.5.conv3
// Shape: [512, 128, 1, 1]
// Memory: 32.0 KB
#define BACKBONE_LAYER2_5_CONV3_DDR_OFFSET 0x000DD800
#define BACKBONE_LAYER2_5_CONV3_DDR_SIZE 32768

// DDR Layer 22: backbone.layer2.6.conv1
// Shape: [128, 512, 1, 1]
// Memory: 32.0 KB
#define BACKBONE_LAYER2_6_CONV1_DDR_OFFSET 0x000E5800
#define BACKBONE_LAYER2_6_CONV1_DDR_SIZE 32768

// DDR Layer 23: backbone.layer2.6.conv2
// Shape: [128, 128, 3, 3]
// Memory: 72.0 KB
#define BACKBONE_LAYER2_6_CONV2_DDR_OFFSET 0x000ED800
#define BACKBONE_LAYER2_6_CONV2_DDR_SIZE 73728

// DDR Layer 24: backbone.layer2.6.conv3
// Shape: [512, 128, 1, 1]
// Memory: 32.0 KB
#define BACKBONE_LAYER2_6_CONV3_DDR_OFFSET 0x000FF800
#define BACKBONE_LAYER2_6_CONV3_DDR_SIZE 32768

// DDR Layer 25: backbone.layer2.7.conv1
// Shape: [128, 512, 1, 1]
// Memory: 32.0 KB
#define BACKBONE_LAYER2_7_CONV1_DDR_OFFSET 0x00107800
#define BACKBONE_LAYER2_7_CONV1_DDR_SIZE 32768

// DDR Layer 26: backbone.layer2.7.conv2
// Shape: [128, 128, 3, 3]
// Memory: 72.0 KB
#define BACKBONE_LAYER2_7_CONV2_DDR_OFFSET 0x0010F800
#define BACKBONE_LAYER2_7_CONV2_DDR_SIZE 73728

// DDR Layer 27: backbone.layer2.7.conv3
// Shape: [512, 128, 1, 1]
// Memory: 32.0 KB
#define BACKBONE_LAYER2_7_CONV3_DDR_OFFSET 0x00121800
#define BACKBONE_LAYER2_7_CONV3_DDR_SIZE 32768

// DDR Layer 28: backbone.layer3.0.conv1
// Shape: [256, 512, 1, 1]
// Memory: 64.0 KB
#define BACKBONE_LAYER3_0_CONV1_DDR_OFFSET 0x00129800
#define BACKBONE_LAYER3_0_CONV1_DDR_SIZE 65536

// DDR Layer 29: classifier.13
// Shape: [256, 512]
// Memory: 64.0 KB
#define CLASSIFIER_13_DDR_OFFSET 0x00139800
#define CLASSIFIER_13_DDR_SIZE 65536

// DDR3 Summary
#define NUM_DDR_LAYERS 30
#define TOTAL_DDR_SIZE_KB 1318.0
#define DDR_BASE_ADDRESS 0x00000000

const ddr_layer_info_t ddr_layer_info[NUM_DDR_LAYERS] = {
    {0x00000000, 18432, 0.003317f, {64, 64, 3, 3}},  // backbone.layer1.0.conv2
    {0x00004800, 18432, 0.003180f, {64, 64, 3, 3}},  // backbone.layer1.1.conv2
    {0x00009000, 18432, 0.002406f, {64, 64, 3, 3}},  // backbone.layer1.2.conv2
    {0x0000D800, 16384, 0.002212f, {128, 256, 1, 1}},  // backbone.layer2.0.conv1
    {0x00011800, 73728, 0.001254f, {128, 128, 3, 3}},  // backbone.layer2.0.conv2
    {0x00023800, 32768, 0.002280f, {512, 128, 1, 1}},  // backbone.layer2.0.conv3
    {0x0002B800, 65536, 0.005348f, {512, 256, 1, 1}},  // backbone.layer2.0.downsample.0
    {0x0003B800, 32768, 0.001655f, {128, 512, 1, 1}},  // backbone.layer2.1.conv1
    {0x00043800, 73728, 0.002730f, {128, 128, 3, 3}},  // backbone.layer2.1.conv2
    {0x00055800, 32768, 0.002345f, {512, 128, 1, 1}},  // backbone.layer2.1.conv3
    {0x0005D800, 32768, 0.001356f, {128, 512, 1, 1}},  // backbone.layer2.2.conv1
    {0x00065800, 73728, 0.001378f, {128, 128, 3, 3}},  // backbone.layer2.2.conv2
    {0x00077800, 32768, 0.001729f, {512, 128, 1, 1}},  // backbone.layer2.2.conv3
    {0x0007F800, 32768, 0.002560f, {128, 512, 1, 1}},  // backbone.layer2.3.conv1
    {0x00087800, 73728, 0.001507f, {128, 128, 3, 3}},  // backbone.layer2.3.conv2
    {0x00099800, 32768, 0.001687f, {512, 128, 1, 1}},  // backbone.layer2.3.conv3
    {0x000A1800, 32768, 0.001490f, {128, 512, 1, 1}},  // backbone.layer2.4.conv1
    {0x000A9800, 73728, 0.001331f, {128, 128, 3, 3}},  // backbone.layer2.4.conv2
    {0x000BB800, 32768, 0.001971f, {512, 128, 1, 1}},  // backbone.layer2.4.conv3
    {0x000C3800, 32768, 0.001688f, {128, 512, 1, 1}},  // backbone.layer2.5.conv1
    {0x000CB800, 73728, 0.001046f, {128, 128, 3, 3}},  // backbone.layer2.5.conv2
    {0x000DD800, 32768, 0.002541f, {512, 128, 1, 1}},  // backbone.layer2.5.conv3
    {0x000E5800, 32768, 0.001279f, {128, 512, 1, 1}},  // backbone.layer2.6.conv1
    {0x000ED800, 73728, 0.001076f, {128, 128, 3, 3}},  // backbone.layer2.6.conv2
    {0x000FF800, 32768, 0.001724f, {512, 128, 1, 1}},  // backbone.layer2.6.conv3
    {0x00107800, 32768, 0.001819f, {128, 512, 1, 1}},  // backbone.layer2.7.conv1
    {0x0010F800, 73728, 0.001558f, {128, 128, 3, 3}},  // backbone.layer2.7.conv2
    {0x00121800, 32768, 0.002719f, {512, 128, 1, 1}},  // backbone.layer2.7.conv3
    {0x00129800, 65536, 0.002517f, {256, 512, 1, 1}},  // backbone.layer3.0.conv1
    {0x00139800, 65536, 0.000246f, {256, 512, 1, 1}}  // classifier.13
};

#endif // DDR_LAYERS_H
