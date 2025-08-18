/*
 * CNN Layer Configuration
 * Main configuration file for HLS CNN accelerator
 * Generated for Zybo Z7-20 FPGA
 */

#ifndef LAYER_CONFIG_H
#define LAYER_CONFIG_H

#include "bram_layers.h"
#include "ddr_layers.h"
#include "streaming_layers.h"

// Overall Model Configuration
#define TOTAL_LAYERS 319
#define NUM_CLASSES 18
#define INPUT_SIZE 224
#define MODEL_MEMORY_KB 31749.7

// Memory Strategy
#define STRATEGY_BRAM_LAYERS 168
#define STRATEGY_DDR_LAYERS 30
#define STRATEGY_STREAMING_LAYERS 121

// Performance Targets
#define TARGET_CLOCK_MHZ 200
#define TARGET_LATENCY_CYCLES 1000
#define MAX_PARALLEL_OPS 16
#define AXI_DATA_WIDTH 512

typedef enum {
    LAYER_TYPE_CONV2D,
    LAYER_TYPE_RELU,
    LAYER_TYPE_POOL,
    LAYER_TYPE_FC,
    LAYER_TYPE_CLASSIFIER
} layer_type_t;

typedef enum {
    MEMORY_TYPE_BRAM,
    MEMORY_TYPE_DDR,
    MEMORY_TYPE_STREAM
} memory_type_t;

#endif // LAYER_CONFIG_H
