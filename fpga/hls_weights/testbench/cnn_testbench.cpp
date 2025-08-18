/*
 * HLS CNN Accelerator Testbench
 * Verifies CNN inference with sample input
 */

#include <iostream>
#include <cmath>
#include "../hls_headers/layer_config.h"
#include "test_input.h"

// Forward declaration of CNN accelerator
void cnn_accelerator(
    ap_uint<8> input_data[TOTAL_INPUT_SIZE],
    float output_data[NUM_CLASSES]
);

int main() {
    std::cout << "Starting CNN Accelerator Testbench...\n";

    // Prepare input data
    ap_uint<8> input_buffer[TOTAL_INPUT_SIZE];
    float output_buffer[NUM_CLASSES];

    // Copy test input
    for (int i = 0; i < TOTAL_INPUT_SIZE; i++) {
        input_buffer[i] = test_input_data[i];
    }

    std::cout << "Running CNN inference...\n";
    cnn_accelerator(input_buffer, output_buffer);

    // Print results
    std::cout << "Output probabilities:\n";
    for (int i = 0; i < NUM_CLASSES; i++) {
        std::cout << "Class " << i << ": " << output_buffer[i] << "\n";
    }

    // Find max prediction
    int max_class = 0;
    float max_prob = output_buffer[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output_buffer[i] > max_prob) {
            max_prob = output_buffer[i];
            max_class = i;
        }
    }

    std::cout << "Predicted class: " << max_class << " (confidence: " << max_prob << ")\n";
    std::cout << "Testbench completed successfully!\n";

    return 0;
}
