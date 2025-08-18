/*
 * BRAM-Stored CNN Layers
 * Fast access layers stored in FPGA BRAM
 * Generated for HLS CNN accelerator
 */

#ifndef BRAM_LAYERS_H
#define BRAM_LAYERS_H

#include <ap_int.h>
#include <ap_fixed.h>

// BRAM Layer: backbone.bn1
// Shape: [64]
// Memory: 0.0 KB
#define BACKBONE_BN1_SIZE 32
extern const ap_uint<8> backbone_bn1_weights[32];
extern const float backbone_bn1_scale;

// BRAM Layer: backbone.layer1.0.bn1
// Shape: [64]
// Memory: 0.0 KB
#define BACKBONE_LAYER1_0_BN1_SIZE 32
extern const ap_uint<8> backbone_layer1_0_bn1_weights[32];
extern const float backbone_layer1_0_bn1_scale;

// BRAM Layer: backbone.layer1.0.bn2
// Shape: [64]
// Memory: 0.0 KB
#define BACKBONE_LAYER1_0_BN2_SIZE 32
extern const ap_uint<8> backbone_layer1_0_bn2_weights[32];
extern const float backbone_layer1_0_bn2_scale;

// BRAM Layer: backbone.layer1.1.bn1
// Shape: [64]
// Memory: 0.0 KB
#define BACKBONE_LAYER1_1_BN1_SIZE 32
extern const ap_uint<8> backbone_layer1_1_bn1_weights[32];
extern const float backbone_layer1_1_bn1_scale;

// BRAM Layer: backbone.layer1.1.bn2
// Shape: [64]
// Memory: 0.0 KB
#define BACKBONE_LAYER1_1_BN2_SIZE 32
extern const ap_uint<8> backbone_layer1_1_bn2_weights[32];
extern const float backbone_layer1_1_bn2_scale;

// BRAM Layer: backbone.layer1.2.bn1
// Shape: [64]
// Memory: 0.0 KB
#define BACKBONE_LAYER1_2_BN1_SIZE 32
extern const ap_uint<8> backbone_layer1_2_bn1_weights[32];
extern const float backbone_layer1_2_bn1_scale;

// BRAM Layer: backbone.layer1.2.bn2
// Shape: [64]
// Memory: 0.0 KB
#define BACKBONE_LAYER1_2_BN2_SIZE 32
extern const ap_uint<8> backbone_layer1_2_bn2_weights[32];
extern const float backbone_layer1_2_bn2_scale;

// BRAM Layer: backbone.layer2.0.bn1
// Shape: [128]
// Memory: 0.1 KB
#define BACKBONE_LAYER2_0_BN1_SIZE 64
extern const ap_uint<8> backbone_layer2_0_bn1_weights[64];
extern const float backbone_layer2_0_bn1_scale;

// BRAM Layer: backbone.layer2.0.bn2
// Shape: [128]
// Memory: 0.1 KB
#define BACKBONE_LAYER2_0_BN2_SIZE 64
extern const ap_uint<8> backbone_layer2_0_bn2_weights[64];
extern const float backbone_layer2_0_bn2_scale;

// BRAM Layer: backbone.layer2.1.bn1
// Shape: [128]
// Memory: 0.1 KB
#define BACKBONE_LAYER2_1_BN1_SIZE 64
extern const ap_uint<8> backbone_layer2_1_bn1_weights[64];
extern const float backbone_layer2_1_bn1_scale;

// BRAM Layer: backbone.layer2.1.bn2
// Shape: [128]
// Memory: 0.1 KB
#define BACKBONE_LAYER2_1_BN2_SIZE 64
extern const ap_uint<8> backbone_layer2_1_bn2_weights[64];
extern const float backbone_layer2_1_bn2_scale;

// BRAM Layer: backbone.layer2.2.bn1
// Shape: [128]
// Memory: 0.1 KB
#define BACKBONE_LAYER2_2_BN1_SIZE 64
extern const ap_uint<8> backbone_layer2_2_bn1_weights[64];
extern const float backbone_layer2_2_bn1_scale;

// BRAM Layer: backbone.layer2.2.bn2
// Shape: [128]
// Memory: 0.1 KB
#define BACKBONE_LAYER2_2_BN2_SIZE 64
extern const ap_uint<8> backbone_layer2_2_bn2_weights[64];
extern const float backbone_layer2_2_bn2_scale;

// BRAM Layer: backbone.layer2.3.bn1
// Shape: [128]
// Memory: 0.1 KB
#define BACKBONE_LAYER2_3_BN1_SIZE 64
extern const ap_uint<8> backbone_layer2_3_bn1_weights[64];
extern const float backbone_layer2_3_bn1_scale;

// BRAM Layer: backbone.layer2.3.bn2
// Shape: [128]
// Memory: 0.1 KB
#define BACKBONE_LAYER2_3_BN2_SIZE 64
extern const ap_uint<8> backbone_layer2_3_bn2_weights[64];
extern const float backbone_layer2_3_bn2_scale;

// BRAM Layer: backbone.layer2.4.bn1
// Shape: [128]
// Memory: 0.1 KB
#define BACKBONE_LAYER2_4_BN1_SIZE 64
extern const ap_uint<8> backbone_layer2_4_bn1_weights[64];
extern const float backbone_layer2_4_bn1_scale;

// BRAM Layer: backbone.layer2.4.bn2
// Shape: [128]
// Memory: 0.1 KB
#define BACKBONE_LAYER2_4_BN2_SIZE 64
extern const ap_uint<8> backbone_layer2_4_bn2_weights[64];
extern const float backbone_layer2_4_bn2_scale;

// BRAM Layer: backbone.layer2.5.bn1
// Shape: [128]
// Memory: 0.1 KB
#define BACKBONE_LAYER2_5_BN1_SIZE 64
extern const ap_uint<8> backbone_layer2_5_bn1_weights[64];
extern const float backbone_layer2_5_bn1_scale;

// BRAM Layer: backbone.layer2.5.bn2
// Shape: [128]
// Memory: 0.1 KB
#define BACKBONE_LAYER2_5_BN2_SIZE 64
extern const ap_uint<8> backbone_layer2_5_bn2_weights[64];
extern const float backbone_layer2_5_bn2_scale;

// BRAM Layer: backbone.layer2.6.bn1
// Shape: [128]
// Memory: 0.1 KB
#define BACKBONE_LAYER2_6_BN1_SIZE 64
extern const ap_uint<8> backbone_layer2_6_bn1_weights[64];
extern const float backbone_layer2_6_bn1_scale;

// BRAM Layer: backbone.layer2.6.bn2
// Shape: [128]
// Memory: 0.1 KB
#define BACKBONE_LAYER2_6_BN2_SIZE 64
extern const ap_uint<8> backbone_layer2_6_bn2_weights[64];
extern const float backbone_layer2_6_bn2_scale;

// BRAM Layer: backbone.layer2.7.bn1
// Shape: [128]
// Memory: 0.1 KB
#define BACKBONE_LAYER2_7_BN1_SIZE 64
extern const ap_uint<8> backbone_layer2_7_bn1_weights[64];
extern const float backbone_layer2_7_bn1_scale;

// BRAM Layer: backbone.layer2.7.bn2
// Shape: [128]
// Memory: 0.1 KB
#define BACKBONE_LAYER2_7_BN2_SIZE 64
extern const ap_uint<8> backbone_layer2_7_bn2_weights[64];
extern const float backbone_layer2_7_bn2_scale;

// BRAM Layer: backbone.layer1.0.bn3
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER1_0_BN3_SIZE 128
extern const ap_uint<8> backbone_layer1_0_bn3_weights[128];
extern const float backbone_layer1_0_bn3_scale;

// BRAM Layer: backbone.layer1.0.downsample.1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER1_0_DOWNSAMPLE_1_SIZE 128
extern const ap_uint<8> backbone_layer1_0_downsample_1_weights[128];
extern const float backbone_layer1_0_downsample_1_scale;

// BRAM Layer: backbone.layer1.1.bn3
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER1_1_BN3_SIZE 128
extern const ap_uint<8> backbone_layer1_1_bn3_weights[128];
extern const float backbone_layer1_1_bn3_scale;

// BRAM Layer: backbone.layer1.2.bn3
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER1_2_BN3_SIZE 128
extern const ap_uint<8> backbone_layer1_2_bn3_weights[128];
extern const float backbone_layer1_2_bn3_scale;

// BRAM Layer: backbone.layer3.0.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_0_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_0_bn1_weights[128];
extern const float backbone_layer3_0_bn1_scale;

// BRAM Layer: backbone.layer3.0.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_0_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_0_bn2_weights[128];
extern const float backbone_layer3_0_bn2_scale;

// BRAM Layer: backbone.layer3.1.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_1_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_1_bn1_weights[128];
extern const float backbone_layer3_1_bn1_scale;

// BRAM Layer: backbone.layer3.1.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_1_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_1_bn2_weights[128];
extern const float backbone_layer3_1_bn2_scale;

// BRAM Layer: backbone.layer3.2.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_2_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_2_bn1_weights[128];
extern const float backbone_layer3_2_bn1_scale;

// BRAM Layer: backbone.layer3.2.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_2_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_2_bn2_weights[128];
extern const float backbone_layer3_2_bn2_scale;

// BRAM Layer: backbone.layer3.3.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_3_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_3_bn1_weights[128];
extern const float backbone_layer3_3_bn1_scale;

// BRAM Layer: backbone.layer3.3.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_3_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_3_bn2_weights[128];
extern const float backbone_layer3_3_bn2_scale;

// BRAM Layer: backbone.layer3.4.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_4_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_4_bn1_weights[128];
extern const float backbone_layer3_4_bn1_scale;

// BRAM Layer: backbone.layer3.4.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_4_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_4_bn2_weights[128];
extern const float backbone_layer3_4_bn2_scale;

// BRAM Layer: backbone.layer3.5.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_5_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_5_bn1_weights[128];
extern const float backbone_layer3_5_bn1_scale;

// BRAM Layer: backbone.layer3.5.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_5_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_5_bn2_weights[128];
extern const float backbone_layer3_5_bn2_scale;

// BRAM Layer: backbone.layer3.6.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_6_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_6_bn1_weights[128];
extern const float backbone_layer3_6_bn1_scale;

// BRAM Layer: backbone.layer3.6.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_6_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_6_bn2_weights[128];
extern const float backbone_layer3_6_bn2_scale;

// BRAM Layer: backbone.layer3.7.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_7_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_7_bn1_weights[128];
extern const float backbone_layer3_7_bn1_scale;

// BRAM Layer: backbone.layer3.7.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_7_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_7_bn2_weights[128];
extern const float backbone_layer3_7_bn2_scale;

// BRAM Layer: backbone.layer3.8.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_8_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_8_bn1_weights[128];
extern const float backbone_layer3_8_bn1_scale;

// BRAM Layer: backbone.layer3.8.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_8_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_8_bn2_weights[128];
extern const float backbone_layer3_8_bn2_scale;

// BRAM Layer: backbone.layer3.9.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_9_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_9_bn1_weights[128];
extern const float backbone_layer3_9_bn1_scale;

// BRAM Layer: backbone.layer3.9.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_9_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_9_bn2_weights[128];
extern const float backbone_layer3_9_bn2_scale;

// BRAM Layer: backbone.layer3.10.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_10_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_10_bn1_weights[128];
extern const float backbone_layer3_10_bn1_scale;

// BRAM Layer: backbone.layer3.10.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_10_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_10_bn2_weights[128];
extern const float backbone_layer3_10_bn2_scale;

// BRAM Layer: backbone.layer3.11.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_11_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_11_bn1_weights[128];
extern const float backbone_layer3_11_bn1_scale;

// BRAM Layer: backbone.layer3.11.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_11_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_11_bn2_weights[128];
extern const float backbone_layer3_11_bn2_scale;

// BRAM Layer: backbone.layer3.12.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_12_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_12_bn1_weights[128];
extern const float backbone_layer3_12_bn1_scale;

// BRAM Layer: backbone.layer3.12.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_12_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_12_bn2_weights[128];
extern const float backbone_layer3_12_bn2_scale;

// BRAM Layer: backbone.layer3.13.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_13_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_13_bn1_weights[128];
extern const float backbone_layer3_13_bn1_scale;

// BRAM Layer: backbone.layer3.13.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_13_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_13_bn2_weights[128];
extern const float backbone_layer3_13_bn2_scale;

// BRAM Layer: backbone.layer3.14.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_14_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_14_bn1_weights[128];
extern const float backbone_layer3_14_bn1_scale;

// BRAM Layer: backbone.layer3.14.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_14_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_14_bn2_weights[128];
extern const float backbone_layer3_14_bn2_scale;

// BRAM Layer: backbone.layer3.15.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_15_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_15_bn1_weights[128];
extern const float backbone_layer3_15_bn1_scale;

// BRAM Layer: backbone.layer3.15.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_15_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_15_bn2_weights[128];
extern const float backbone_layer3_15_bn2_scale;

// BRAM Layer: backbone.layer3.16.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_16_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_16_bn1_weights[128];
extern const float backbone_layer3_16_bn1_scale;

// BRAM Layer: backbone.layer3.16.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_16_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_16_bn2_weights[128];
extern const float backbone_layer3_16_bn2_scale;

// BRAM Layer: backbone.layer3.17.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_17_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_17_bn1_weights[128];
extern const float backbone_layer3_17_bn1_scale;

// BRAM Layer: backbone.layer3.17.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_17_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_17_bn2_weights[128];
extern const float backbone_layer3_17_bn2_scale;

// BRAM Layer: backbone.layer3.18.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_18_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_18_bn1_weights[128];
extern const float backbone_layer3_18_bn1_scale;

// BRAM Layer: backbone.layer3.18.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_18_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_18_bn2_weights[128];
extern const float backbone_layer3_18_bn2_scale;

// BRAM Layer: backbone.layer3.19.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_19_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_19_bn1_weights[128];
extern const float backbone_layer3_19_bn1_scale;

// BRAM Layer: backbone.layer3.19.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_19_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_19_bn2_weights[128];
extern const float backbone_layer3_19_bn2_scale;

// BRAM Layer: backbone.layer3.20.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_20_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_20_bn1_weights[128];
extern const float backbone_layer3_20_bn1_scale;

// BRAM Layer: backbone.layer3.20.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_20_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_20_bn2_weights[128];
extern const float backbone_layer3_20_bn2_scale;

// BRAM Layer: backbone.layer3.21.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_21_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_21_bn1_weights[128];
extern const float backbone_layer3_21_bn1_scale;

// BRAM Layer: backbone.layer3.21.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_21_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_21_bn2_weights[128];
extern const float backbone_layer3_21_bn2_scale;

// BRAM Layer: backbone.layer3.22.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_22_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_22_bn1_weights[128];
extern const float backbone_layer3_22_bn1_scale;

// BRAM Layer: backbone.layer3.22.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_22_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_22_bn2_weights[128];
extern const float backbone_layer3_22_bn2_scale;

// BRAM Layer: backbone.layer3.23.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_23_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_23_bn1_weights[128];
extern const float backbone_layer3_23_bn1_scale;

// BRAM Layer: backbone.layer3.23.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_23_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_23_bn2_weights[128];
extern const float backbone_layer3_23_bn2_scale;

// BRAM Layer: backbone.layer3.24.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_24_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_24_bn1_weights[128];
extern const float backbone_layer3_24_bn1_scale;

// BRAM Layer: backbone.layer3.24.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_24_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_24_bn2_weights[128];
extern const float backbone_layer3_24_bn2_scale;

// BRAM Layer: backbone.layer3.25.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_25_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_25_bn1_weights[128];
extern const float backbone_layer3_25_bn1_scale;

// BRAM Layer: backbone.layer3.25.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_25_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_25_bn2_weights[128];
extern const float backbone_layer3_25_bn2_scale;

// BRAM Layer: backbone.layer3.26.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_26_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_26_bn1_weights[128];
extern const float backbone_layer3_26_bn1_scale;

// BRAM Layer: backbone.layer3.26.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_26_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_26_bn2_weights[128];
extern const float backbone_layer3_26_bn2_scale;

// BRAM Layer: backbone.layer3.27.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_27_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_27_bn1_weights[128];
extern const float backbone_layer3_27_bn1_scale;

// BRAM Layer: backbone.layer3.27.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_27_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_27_bn2_weights[128];
extern const float backbone_layer3_27_bn2_scale;

// BRAM Layer: backbone.layer3.28.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_28_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_28_bn1_weights[128];
extern const float backbone_layer3_28_bn1_scale;

// BRAM Layer: backbone.layer3.28.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_28_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_28_bn2_weights[128];
extern const float backbone_layer3_28_bn2_scale;

// BRAM Layer: backbone.layer3.29.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_29_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_29_bn1_weights[128];
extern const float backbone_layer3_29_bn1_scale;

// BRAM Layer: backbone.layer3.29.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_29_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_29_bn2_weights[128];
extern const float backbone_layer3_29_bn2_scale;

// BRAM Layer: backbone.layer3.30.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_30_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_30_bn1_weights[128];
extern const float backbone_layer3_30_bn1_scale;

// BRAM Layer: backbone.layer3.30.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_30_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_30_bn2_weights[128];
extern const float backbone_layer3_30_bn2_scale;

// BRAM Layer: backbone.layer3.31.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_31_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_31_bn1_weights[128];
extern const float backbone_layer3_31_bn1_scale;

// BRAM Layer: backbone.layer3.31.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_31_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_31_bn2_weights[128];
extern const float backbone_layer3_31_bn2_scale;

// BRAM Layer: backbone.layer3.32.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_32_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_32_bn1_weights[128];
extern const float backbone_layer3_32_bn1_scale;

// BRAM Layer: backbone.layer3.32.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_32_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_32_bn2_weights[128];
extern const float backbone_layer3_32_bn2_scale;

// BRAM Layer: backbone.layer3.33.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_33_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_33_bn1_weights[128];
extern const float backbone_layer3_33_bn1_scale;

// BRAM Layer: backbone.layer3.33.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_33_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_33_bn2_weights[128];
extern const float backbone_layer3_33_bn2_scale;

// BRAM Layer: backbone.layer3.34.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_34_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_34_bn1_weights[128];
extern const float backbone_layer3_34_bn1_scale;

// BRAM Layer: backbone.layer3.34.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_34_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_34_bn2_weights[128];
extern const float backbone_layer3_34_bn2_scale;

// BRAM Layer: backbone.layer3.35.bn1
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_35_BN1_SIZE 128
extern const ap_uint<8> backbone_layer3_35_bn1_weights[128];
extern const float backbone_layer3_35_bn1_scale;

// BRAM Layer: backbone.layer3.35.bn2
// Shape: [256]
// Memory: 0.1 KB
#define BACKBONE_LAYER3_35_BN2_SIZE 128
extern const ap_uint<8> backbone_layer3_35_bn2_weights[128];
extern const float backbone_layer3_35_bn2_scale;

// BRAM Layer: classifier.15
// Shape: [256]
// Memory: 0.1 KB
#define CLASSIFIER_15_SIZE 128
extern const ap_uint<8> classifier_15_weights[128];
extern const float classifier_15_scale;

// BRAM Layer: backbone.layer2.0.bn3
// Shape: [512]
// Memory: 0.2 KB
#define BACKBONE_LAYER2_0_BN3_SIZE 256
extern const ap_uint<8> backbone_layer2_0_bn3_weights[256];
extern const float backbone_layer2_0_bn3_scale;

// BRAM Layer: backbone.layer2.0.downsample.1
// Shape: [512]
// Memory: 0.2 KB
#define BACKBONE_LAYER2_0_DOWNSAMPLE_1_SIZE 256
extern const ap_uint<8> backbone_layer2_0_downsample_1_weights[256];
extern const float backbone_layer2_0_downsample_1_scale;

// BRAM Layer: backbone.layer2.1.bn3
// Shape: [512]
// Memory: 0.2 KB
#define BACKBONE_LAYER2_1_BN3_SIZE 256
extern const ap_uint<8> backbone_layer2_1_bn3_weights[256];
extern const float backbone_layer2_1_bn3_scale;

// BRAM Layer: backbone.layer2.2.bn3
// Shape: [512]
// Memory: 0.2 KB
#define BACKBONE_LAYER2_2_BN3_SIZE 256
extern const ap_uint<8> backbone_layer2_2_bn3_weights[256];
extern const float backbone_layer2_2_bn3_scale;

// BRAM Layer: backbone.layer2.3.bn3
// Shape: [512]
// Memory: 0.2 KB
#define BACKBONE_LAYER2_3_BN3_SIZE 256
extern const ap_uint<8> backbone_layer2_3_bn3_weights[256];
extern const float backbone_layer2_3_bn3_scale;

// BRAM Layer: backbone.layer2.4.bn3
// Shape: [512]
// Memory: 0.2 KB
#define BACKBONE_LAYER2_4_BN3_SIZE 256
extern const ap_uint<8> backbone_layer2_4_bn3_weights[256];
extern const float backbone_layer2_4_bn3_scale;

// BRAM Layer: backbone.layer2.5.bn3
// Shape: [512]
// Memory: 0.2 KB
#define BACKBONE_LAYER2_5_BN3_SIZE 256
extern const ap_uint<8> backbone_layer2_5_bn3_weights[256];
extern const float backbone_layer2_5_bn3_scale;

// BRAM Layer: backbone.layer2.6.bn3
// Shape: [512]
// Memory: 0.2 KB
#define BACKBONE_LAYER2_6_BN3_SIZE 256
extern const ap_uint<8> backbone_layer2_6_bn3_weights[256];
extern const float backbone_layer2_6_bn3_scale;

// BRAM Layer: backbone.layer2.7.bn3
// Shape: [512]
// Memory: 0.2 KB
#define BACKBONE_LAYER2_7_BN3_SIZE 256
extern const ap_uint<8> backbone_layer2_7_bn3_weights[256];
extern const float backbone_layer2_7_bn3_scale;

// BRAM Layer: backbone.layer4.0.bn1
// Shape: [512]
// Memory: 0.2 KB
#define BACKBONE_LAYER4_0_BN1_SIZE 256
extern const ap_uint<8> backbone_layer4_0_bn1_weights[256];
extern const float backbone_layer4_0_bn1_scale;

// BRAM Layer: backbone.layer4.0.bn2
// Shape: [512]
// Memory: 0.2 KB
#define BACKBONE_LAYER4_0_BN2_SIZE 256
extern const ap_uint<8> backbone_layer4_0_bn2_weights[256];
extern const float backbone_layer4_0_bn2_scale;

// BRAM Layer: backbone.layer4.1.bn1
// Shape: [512]
// Memory: 0.2 KB
#define BACKBONE_LAYER4_1_BN1_SIZE 256
extern const ap_uint<8> backbone_layer4_1_bn1_weights[256];
extern const float backbone_layer4_1_bn1_scale;

// BRAM Layer: backbone.layer4.1.bn2
// Shape: [512]
// Memory: 0.2 KB
#define BACKBONE_LAYER4_1_BN2_SIZE 256
extern const ap_uint<8> backbone_layer4_1_bn2_weights[256];
extern const float backbone_layer4_1_bn2_scale;

// BRAM Layer: backbone.layer4.2.bn1
// Shape: [512]
// Memory: 0.2 KB
#define BACKBONE_LAYER4_2_BN1_SIZE 256
extern const ap_uint<8> backbone_layer4_2_bn1_weights[256];
extern const float backbone_layer4_2_bn1_scale;

// BRAM Layer: backbone.layer4.2.bn2
// Shape: [512]
// Memory: 0.2 KB
#define BACKBONE_LAYER4_2_BN2_SIZE 256
extern const ap_uint<8> backbone_layer4_2_bn2_weights[256];
extern const float backbone_layer4_2_bn2_scale;

// BRAM Layer: classifier.11
// Shape: [512]
// Memory: 0.2 KB
#define CLASSIFIER_11_SIZE 256
extern const ap_uint<8> classifier_11_weights[256];
extern const float classifier_11_scale;

// BRAM Layer: backbone.layer3.0.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_0_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_0_bn3_weights[512];
extern const float backbone_layer3_0_bn3_scale;

// BRAM Layer: backbone.layer3.0.downsample.1
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_0_DOWNSAMPLE_1_SIZE 512
extern const ap_uint<8> backbone_layer3_0_downsample_1_weights[512];
extern const float backbone_layer3_0_downsample_1_scale;

// BRAM Layer: backbone.layer3.1.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_1_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_1_bn3_weights[512];
extern const float backbone_layer3_1_bn3_scale;

// BRAM Layer: backbone.layer3.2.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_2_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_2_bn3_weights[512];
extern const float backbone_layer3_2_bn3_scale;

// BRAM Layer: backbone.layer3.3.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_3_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_3_bn3_weights[512];
extern const float backbone_layer3_3_bn3_scale;

// BRAM Layer: backbone.layer3.4.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_4_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_4_bn3_weights[512];
extern const float backbone_layer3_4_bn3_scale;

// BRAM Layer: backbone.layer3.5.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_5_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_5_bn3_weights[512];
extern const float backbone_layer3_5_bn3_scale;

// BRAM Layer: backbone.layer3.6.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_6_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_6_bn3_weights[512];
extern const float backbone_layer3_6_bn3_scale;

// BRAM Layer: backbone.layer3.7.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_7_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_7_bn3_weights[512];
extern const float backbone_layer3_7_bn3_scale;

// BRAM Layer: backbone.layer3.8.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_8_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_8_bn3_weights[512];
extern const float backbone_layer3_8_bn3_scale;

// BRAM Layer: backbone.layer3.9.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_9_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_9_bn3_weights[512];
extern const float backbone_layer3_9_bn3_scale;

// BRAM Layer: backbone.layer3.10.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_10_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_10_bn3_weights[512];
extern const float backbone_layer3_10_bn3_scale;

// BRAM Layer: backbone.layer3.11.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_11_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_11_bn3_weights[512];
extern const float backbone_layer3_11_bn3_scale;

// BRAM Layer: backbone.layer3.12.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_12_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_12_bn3_weights[512];
extern const float backbone_layer3_12_bn3_scale;

// BRAM Layer: backbone.layer3.13.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_13_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_13_bn3_weights[512];
extern const float backbone_layer3_13_bn3_scale;

// BRAM Layer: backbone.layer3.14.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_14_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_14_bn3_weights[512];
extern const float backbone_layer3_14_bn3_scale;

// BRAM Layer: backbone.layer3.15.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_15_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_15_bn3_weights[512];
extern const float backbone_layer3_15_bn3_scale;

// BRAM Layer: backbone.layer3.16.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_16_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_16_bn3_weights[512];
extern const float backbone_layer3_16_bn3_scale;

// BRAM Layer: backbone.layer3.17.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_17_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_17_bn3_weights[512];
extern const float backbone_layer3_17_bn3_scale;

// BRAM Layer: backbone.layer3.18.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_18_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_18_bn3_weights[512];
extern const float backbone_layer3_18_bn3_scale;

// BRAM Layer: backbone.layer3.19.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_19_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_19_bn3_weights[512];
extern const float backbone_layer3_19_bn3_scale;

// BRAM Layer: backbone.layer3.20.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_20_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_20_bn3_weights[512];
extern const float backbone_layer3_20_bn3_scale;

// BRAM Layer: backbone.layer3.21.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_21_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_21_bn3_weights[512];
extern const float backbone_layer3_21_bn3_scale;

// BRAM Layer: backbone.layer3.22.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_22_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_22_bn3_weights[512];
extern const float backbone_layer3_22_bn3_scale;

// BRAM Layer: backbone.layer3.23.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_23_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_23_bn3_weights[512];
extern const float backbone_layer3_23_bn3_scale;

// BRAM Layer: backbone.layer3.24.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_24_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_24_bn3_weights[512];
extern const float backbone_layer3_24_bn3_scale;

// BRAM Layer: backbone.layer3.25.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_25_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_25_bn3_weights[512];
extern const float backbone_layer3_25_bn3_scale;

// BRAM Layer: backbone.layer3.26.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_26_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_26_bn3_weights[512];
extern const float backbone_layer3_26_bn3_scale;

// BRAM Layer: backbone.layer3.27.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_27_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_27_bn3_weights[512];
extern const float backbone_layer3_27_bn3_scale;

// BRAM Layer: backbone.layer3.28.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_28_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_28_bn3_weights[512];
extern const float backbone_layer3_28_bn3_scale;

// BRAM Layer: backbone.layer3.29.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_29_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_29_bn3_weights[512];
extern const float backbone_layer3_29_bn3_scale;

// BRAM Layer: backbone.layer3.30.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_30_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_30_bn3_weights[512];
extern const float backbone_layer3_30_bn3_scale;

// BRAM Layer: backbone.layer3.31.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_31_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_31_bn3_weights[512];
extern const float backbone_layer3_31_bn3_scale;

// BRAM Layer: backbone.layer3.32.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_32_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_32_bn3_weights[512];
extern const float backbone_layer3_32_bn3_scale;

// BRAM Layer: backbone.layer3.33.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_33_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_33_bn3_weights[512];
extern const float backbone_layer3_33_bn3_scale;

// BRAM Layer: backbone.layer3.34.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_34_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_34_bn3_weights[512];
extern const float backbone_layer3_34_bn3_scale;

// BRAM Layer: backbone.layer3.35.bn3
// Shape: [1024]
// Memory: 0.5 KB
#define BACKBONE_LAYER3_35_BN3_SIZE 512
extern const ap_uint<8> backbone_layer3_35_bn3_weights[512];
extern const float backbone_layer3_35_bn3_scale;

// BRAM Layer: classifier.7
// Shape: [1024]
// Memory: 0.5 KB
#define CLASSIFIER_7_SIZE 512
extern const ap_uint<8> classifier_7_weights[512];
extern const float classifier_7_scale;

// BRAM Layer: backbone.layer4.0.bn3
// Shape: [2048]
// Memory: 1.0 KB
#define BACKBONE_LAYER4_0_BN3_SIZE 1024
extern const ap_uint<8> backbone_layer4_0_bn3_weights[1024];
extern const float backbone_layer4_0_bn3_scale;

// BRAM Layer: backbone.layer4.0.downsample.1
// Shape: [2048]
// Memory: 1.0 KB
#define BACKBONE_LAYER4_0_DOWNSAMPLE_1_SIZE 1024
extern const ap_uint<8> backbone_layer4_0_downsample_1_weights[1024];
extern const float backbone_layer4_0_downsample_1_scale;

// BRAM Layer: backbone.layer4.1.bn3
// Shape: [2048]
// Memory: 1.0 KB
#define BACKBONE_LAYER4_1_BN3_SIZE 1024
extern const ap_uint<8> backbone_layer4_1_bn3_weights[1024];
extern const float backbone_layer4_1_bn3_scale;

// BRAM Layer: backbone.layer4.2.bn3
// Shape: [2048]
// Memory: 1.0 KB
#define BACKBONE_LAYER4_2_BN3_SIZE 1024
extern const ap_uint<8> backbone_layer4_2_bn3_weights[1024];
extern const float backbone_layer4_2_bn3_scale;

// BRAM Layer: classifier.3
// Shape: [2048]
// Memory: 1.0 KB
#define CLASSIFIER_3_SIZE 1024
extern const ap_uint<8> classifier_3_weights[1024];
extern const float classifier_3_scale;

// BRAM Layer: backbone.layer1.0.conv1
// Shape: [64, 64, 1, 1]
// Memory: 2.0 KB
#define BACKBONE_LAYER1_0_CONV1_SIZE 2048
extern const ap_uint<8> backbone_layer1_0_conv1_weights[2048];
extern const float backbone_layer1_0_conv1_scale;

// BRAM Layer: classifier.17
// Shape: [18, 256]
// Memory: 2.2 KB
#define CLASSIFIER_17_SIZE 2304
extern const ap_uint<8> classifier_17_weights[2304];
extern const float classifier_17_scale;

// BRAM Layer: backbone.conv1
// Shape: [64, 3, 7, 7]
// Memory: 4.6 KB
#define BACKBONE_CONV1_SIZE 4704
extern const ap_uint<8> backbone_conv1_weights[4704];
extern const float backbone_conv1_scale;

// BRAM Layer: backbone.layer1.0.conv3
// Shape: [256, 64, 1, 1]
// Memory: 8.0 KB
#define BACKBONE_LAYER1_0_CONV3_SIZE 8192
extern const ap_uint<8> backbone_layer1_0_conv3_weights[8192];
extern const float backbone_layer1_0_conv3_scale;

// BRAM Layer: backbone.layer1.0.downsample.0
// Shape: [256, 64, 1, 1]
// Memory: 8.0 KB
#define BACKBONE_LAYER1_0_DOWNSAMPLE_0_SIZE 8192
extern const ap_uint<8> backbone_layer1_0_downsample_0_weights[8192];
extern const float backbone_layer1_0_downsample_0_scale;

// BRAM Layer: backbone.layer1.1.conv1
// Shape: [64, 256, 1, 1]
// Memory: 8.0 KB
#define BACKBONE_LAYER1_1_CONV1_SIZE 8192
extern const ap_uint<8> backbone_layer1_1_conv1_weights[8192];
extern const float backbone_layer1_1_conv1_scale;

// BRAM Layer: backbone.layer1.1.conv3
// Shape: [256, 64, 1, 1]
// Memory: 8.0 KB
#define BACKBONE_LAYER1_1_CONV3_SIZE 8192
extern const ap_uint<8> backbone_layer1_1_conv3_weights[8192];
extern const float backbone_layer1_1_conv3_scale;

// BRAM Layer: backbone.layer1.2.conv1
// Shape: [64, 256, 1, 1]
// Memory: 8.0 KB
#define BACKBONE_LAYER1_2_CONV1_SIZE 8192
extern const ap_uint<8> backbone_layer1_2_conv1_weights[8192];
extern const float backbone_layer1_2_conv1_scale;

// BRAM Layer: backbone.layer1.2.conv3
// Shape: [256, 64, 1, 1]
// Memory: 8.0 KB
#define BACKBONE_LAYER1_2_CONV3_SIZE 8192
extern const ap_uint<8> backbone_layer1_2_conv3_weights[8192];
extern const float backbone_layer1_2_conv3_scale;

// BRAM Summary
#define NUM_BRAM_LAYERS 168
#define TOTAL_BRAM_SIZE_KB 95.7
#define BRAM_UTILIZATION_PERCENT 15.2

#endif // BRAM_LAYERS_H
