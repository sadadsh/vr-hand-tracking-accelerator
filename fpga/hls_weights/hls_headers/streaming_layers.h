/*
 * Streaming CNN Layers
 * Very large layers processed with weight streaming
 * Generated for HLS CNN accelerator
 */

#ifndef STREAMING_LAYERS_H
#define STREAMING_LAYERS_H

#include <ap_int.h>
#include <hls_stream.h>

// Streaming Configuration
#define STREAM_BUFFER_SIZE 64
#define MAX_STREAM_CHUNKS 1024

typedef struct {
    ap_uint<32> total_size;
    ap_uint<16> chunk_size;
    ap_uint<16> num_chunks;
    float scale_factor;
} stream_layer_info_t;

// Streaming Layer 0: backbone.layer3.0.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_0_CONV2_CHUNKS 4609

// Streaming Layer 1: backbone.layer3.0.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_0_CONV3_CHUNKS 2049

// Streaming Layer 2: backbone.layer3.0.downsample.0
// Shape: [1024, 512, 1, 1]
// Memory: 256.0 KB
#define BACKBONE_LAYER3_0_DOWNSAMPLE_0_CHUNKS 4097

// Streaming Layer 3: backbone.layer3.1.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_1_CONV1_CHUNKS 2049

// Streaming Layer 4: backbone.layer3.1.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_1_CONV2_CHUNKS 4609

// Streaming Layer 5: backbone.layer3.1.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_1_CONV3_CHUNKS 2049

// Streaming Layer 6: backbone.layer3.2.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_2_CONV1_CHUNKS 2049

// Streaming Layer 7: backbone.layer3.2.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_2_CONV2_CHUNKS 4609

// Streaming Layer 8: backbone.layer3.2.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_2_CONV3_CHUNKS 2049

// Streaming Layer 9: backbone.layer3.3.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_3_CONV1_CHUNKS 2049

// Streaming Layer 10: backbone.layer3.3.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_3_CONV2_CHUNKS 4609

// Streaming Layer 11: backbone.layer3.3.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_3_CONV3_CHUNKS 2049

// Streaming Layer 12: backbone.layer3.4.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_4_CONV1_CHUNKS 2049

// Streaming Layer 13: backbone.layer3.4.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_4_CONV2_CHUNKS 4609

// Streaming Layer 14: backbone.layer3.4.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_4_CONV3_CHUNKS 2049

// Streaming Layer 15: backbone.layer3.5.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_5_CONV1_CHUNKS 2049

// Streaming Layer 16: backbone.layer3.5.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_5_CONV2_CHUNKS 4609

// Streaming Layer 17: backbone.layer3.5.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_5_CONV3_CHUNKS 2049

// Streaming Layer 18: backbone.layer3.6.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_6_CONV1_CHUNKS 2049

// Streaming Layer 19: backbone.layer3.6.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_6_CONV2_CHUNKS 4609

// Streaming Layer 20: backbone.layer3.6.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_6_CONV3_CHUNKS 2049

// Streaming Layer 21: backbone.layer3.7.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_7_CONV1_CHUNKS 2049

// Streaming Layer 22: backbone.layer3.7.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_7_CONV2_CHUNKS 4609

// Streaming Layer 23: backbone.layer3.7.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_7_CONV3_CHUNKS 2049

// Streaming Layer 24: backbone.layer3.8.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_8_CONV1_CHUNKS 2049

// Streaming Layer 25: backbone.layer3.8.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_8_CONV2_CHUNKS 4609

// Streaming Layer 26: backbone.layer3.8.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_8_CONV3_CHUNKS 2049

// Streaming Layer 27: backbone.layer3.9.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_9_CONV1_CHUNKS 2049

// Streaming Layer 28: backbone.layer3.9.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_9_CONV2_CHUNKS 4609

// Streaming Layer 29: backbone.layer3.9.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_9_CONV3_CHUNKS 2049

// Streaming Layer 30: backbone.layer3.10.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_10_CONV1_CHUNKS 2049

// Streaming Layer 31: backbone.layer3.10.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_10_CONV2_CHUNKS 4609

// Streaming Layer 32: backbone.layer3.10.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_10_CONV3_CHUNKS 2049

// Streaming Layer 33: backbone.layer3.11.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_11_CONV1_CHUNKS 2049

// Streaming Layer 34: backbone.layer3.11.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_11_CONV2_CHUNKS 4609

// Streaming Layer 35: backbone.layer3.11.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_11_CONV3_CHUNKS 2049

// Streaming Layer 36: backbone.layer3.12.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_12_CONV1_CHUNKS 2049

// Streaming Layer 37: backbone.layer3.12.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_12_CONV2_CHUNKS 4609

// Streaming Layer 38: backbone.layer3.12.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_12_CONV3_CHUNKS 2049

// Streaming Layer 39: backbone.layer3.13.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_13_CONV1_CHUNKS 2049

// Streaming Layer 40: backbone.layer3.13.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_13_CONV2_CHUNKS 4609

// Streaming Layer 41: backbone.layer3.13.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_13_CONV3_CHUNKS 2049

// Streaming Layer 42: backbone.layer3.14.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_14_CONV1_CHUNKS 2049

// Streaming Layer 43: backbone.layer3.14.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_14_CONV2_CHUNKS 4609

// Streaming Layer 44: backbone.layer3.14.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_14_CONV3_CHUNKS 2049

// Streaming Layer 45: backbone.layer3.15.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_15_CONV1_CHUNKS 2049

// Streaming Layer 46: backbone.layer3.15.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_15_CONV2_CHUNKS 4609

// Streaming Layer 47: backbone.layer3.15.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_15_CONV3_CHUNKS 2049

// Streaming Layer 48: backbone.layer3.16.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_16_CONV1_CHUNKS 2049

// Streaming Layer 49: backbone.layer3.16.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_16_CONV2_CHUNKS 4609

// Streaming Layer 50: backbone.layer3.16.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_16_CONV3_CHUNKS 2049

// Streaming Layer 51: backbone.layer3.17.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_17_CONV1_CHUNKS 2049

// Streaming Layer 52: backbone.layer3.17.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_17_CONV2_CHUNKS 4609

// Streaming Layer 53: backbone.layer3.17.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_17_CONV3_CHUNKS 2049

// Streaming Layer 54: backbone.layer3.18.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_18_CONV1_CHUNKS 2049

// Streaming Layer 55: backbone.layer3.18.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_18_CONV2_CHUNKS 4609

// Streaming Layer 56: backbone.layer3.18.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_18_CONV3_CHUNKS 2049

// Streaming Layer 57: backbone.layer3.19.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_19_CONV1_CHUNKS 2049

// Streaming Layer 58: backbone.layer3.19.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_19_CONV2_CHUNKS 4609

// Streaming Layer 59: backbone.layer3.19.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_19_CONV3_CHUNKS 2049

// Streaming Layer 60: backbone.layer3.20.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_20_CONV1_CHUNKS 2049

// Streaming Layer 61: backbone.layer3.20.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_20_CONV2_CHUNKS 4609

// Streaming Layer 62: backbone.layer3.20.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_20_CONV3_CHUNKS 2049

// Streaming Layer 63: backbone.layer3.21.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_21_CONV1_CHUNKS 2049

// Streaming Layer 64: backbone.layer3.21.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_21_CONV2_CHUNKS 4609

// Streaming Layer 65: backbone.layer3.21.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_21_CONV3_CHUNKS 2049

// Streaming Layer 66: backbone.layer3.22.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_22_CONV1_CHUNKS 2049

// Streaming Layer 67: backbone.layer3.22.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_22_CONV2_CHUNKS 4609

// Streaming Layer 68: backbone.layer3.22.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_22_CONV3_CHUNKS 2049

// Streaming Layer 69: backbone.layer3.23.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_23_CONV1_CHUNKS 2049

// Streaming Layer 70: backbone.layer3.23.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_23_CONV2_CHUNKS 4609

// Streaming Layer 71: backbone.layer3.23.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_23_CONV3_CHUNKS 2049

// Streaming Layer 72: backbone.layer3.24.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_24_CONV1_CHUNKS 2049

// Streaming Layer 73: backbone.layer3.24.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_24_CONV2_CHUNKS 4609

// Streaming Layer 74: backbone.layer3.24.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_24_CONV3_CHUNKS 2049

// Streaming Layer 75: backbone.layer3.25.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_25_CONV1_CHUNKS 2049

// Streaming Layer 76: backbone.layer3.25.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_25_CONV2_CHUNKS 4609

// Streaming Layer 77: backbone.layer3.25.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_25_CONV3_CHUNKS 2049

// Streaming Layer 78: backbone.layer3.26.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_26_CONV1_CHUNKS 2049

// Streaming Layer 79: backbone.layer3.26.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_26_CONV2_CHUNKS 4609

// Streaming Layer 80: backbone.layer3.26.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_26_CONV3_CHUNKS 2049

// Streaming Layer 81: backbone.layer3.27.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_27_CONV1_CHUNKS 2049

// Streaming Layer 82: backbone.layer3.27.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_27_CONV2_CHUNKS 4609

// Streaming Layer 83: backbone.layer3.27.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_27_CONV3_CHUNKS 2049

// Streaming Layer 84: backbone.layer3.28.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_28_CONV1_CHUNKS 2049

// Streaming Layer 85: backbone.layer3.28.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_28_CONV2_CHUNKS 4609

// Streaming Layer 86: backbone.layer3.28.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_28_CONV3_CHUNKS 2049

// Streaming Layer 87: backbone.layer3.29.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_29_CONV1_CHUNKS 2049

// Streaming Layer 88: backbone.layer3.29.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_29_CONV2_CHUNKS 4609

// Streaming Layer 89: backbone.layer3.29.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_29_CONV3_CHUNKS 2049

// Streaming Layer 90: backbone.layer3.30.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_30_CONV1_CHUNKS 2049

// Streaming Layer 91: backbone.layer3.30.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_30_CONV2_CHUNKS 4609

// Streaming Layer 92: backbone.layer3.30.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_30_CONV3_CHUNKS 2049

// Streaming Layer 93: backbone.layer3.31.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_31_CONV1_CHUNKS 2049

// Streaming Layer 94: backbone.layer3.31.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_31_CONV2_CHUNKS 4609

// Streaming Layer 95: backbone.layer3.31.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_31_CONV3_CHUNKS 2049

// Streaming Layer 96: backbone.layer3.32.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_32_CONV1_CHUNKS 2049

// Streaming Layer 97: backbone.layer3.32.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_32_CONV2_CHUNKS 4609

// Streaming Layer 98: backbone.layer3.32.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_32_CONV3_CHUNKS 2049

// Streaming Layer 99: backbone.layer3.33.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_33_CONV1_CHUNKS 2049

// Streaming Layer 100: backbone.layer3.33.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_33_CONV2_CHUNKS 4609

// Streaming Layer 101: backbone.layer3.33.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_33_CONV3_CHUNKS 2049

// Streaming Layer 102: backbone.layer3.34.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_34_CONV1_CHUNKS 2049

// Streaming Layer 103: backbone.layer3.34.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_34_CONV2_CHUNKS 4609

// Streaming Layer 104: backbone.layer3.34.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_34_CONV3_CHUNKS 2049

// Streaming Layer 105: backbone.layer3.35.conv1
// Shape: [256, 1024, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_35_CONV1_CHUNKS 2049

// Streaming Layer 106: backbone.layer3.35.conv2
// Shape: [256, 256, 3, 3]
// Memory: 288.0 KB
#define BACKBONE_LAYER3_35_CONV2_CHUNKS 4609

// Streaming Layer 107: backbone.layer3.35.conv3
// Shape: [1024, 256, 1, 1]
// Memory: 128.0 KB
#define BACKBONE_LAYER3_35_CONV3_CHUNKS 2049

// Streaming Layer 108: backbone.layer4.0.conv1
// Shape: [512, 1024, 1, 1]
// Memory: 256.0 KB
#define BACKBONE_LAYER4_0_CONV1_CHUNKS 4097

// Streaming Layer 109: backbone.layer4.0.conv2
// Shape: [512, 512, 3, 3]
// Memory: 1152.0 KB
#define BACKBONE_LAYER4_0_CONV2_CHUNKS 18433

// Streaming Layer 110: backbone.layer4.0.conv3
// Shape: [2048, 512, 1, 1]
// Memory: 512.0 KB
#define BACKBONE_LAYER4_0_CONV3_CHUNKS 8193

// Streaming Layer 111: backbone.layer4.0.downsample.0
// Shape: [2048, 1024, 1, 1]
// Memory: 1024.0 KB
#define BACKBONE_LAYER4_0_DOWNSAMPLE_0_CHUNKS 16385

// Streaming Layer 112: backbone.layer4.1.conv1
// Shape: [512, 2048, 1, 1]
// Memory: 512.0 KB
#define BACKBONE_LAYER4_1_CONV1_CHUNKS 8193

// Streaming Layer 113: backbone.layer4.1.conv2
// Shape: [512, 512, 3, 3]
// Memory: 1152.0 KB
#define BACKBONE_LAYER4_1_CONV2_CHUNKS 18433

// Streaming Layer 114: backbone.layer4.1.conv3
// Shape: [2048, 512, 1, 1]
// Memory: 512.0 KB
#define BACKBONE_LAYER4_1_CONV3_CHUNKS 8193

// Streaming Layer 115: backbone.layer4.2.conv1
// Shape: [512, 2048, 1, 1]
// Memory: 512.0 KB
#define BACKBONE_LAYER4_2_CONV1_CHUNKS 8193

// Streaming Layer 116: backbone.layer4.2.conv2
// Shape: [512, 512, 3, 3]
// Memory: 1152.0 KB
#define BACKBONE_LAYER4_2_CONV2_CHUNKS 18433

// Streaming Layer 117: backbone.layer4.2.conv3
// Shape: [2048, 512, 1, 1]
// Memory: 512.0 KB
#define BACKBONE_LAYER4_2_CONV3_CHUNKS 8193

// Streaming Layer 118: classifier.1
// Shape: [2048, 2048]
// Memory: 2048.0 KB
#define CLASSIFIER_1_CHUNKS 32769

// Streaming Layer 119: classifier.5
// Shape: [1024, 2048]
// Memory: 1024.0 KB
#define CLASSIFIER_5_CHUNKS 16385

// Streaming Layer 120: classifier.9
// Shape: [512, 1024]
// Memory: 256.0 KB
#define CLASSIFIER_9_CHUNKS 4097

#define NUM_STREAMING_LAYERS 121
#define TOTAL_STREAMING_SIZE_KB 30336.0

#endif // STREAMING_LAYERS_H
