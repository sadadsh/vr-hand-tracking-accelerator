// ============================================================================
// Convolution Layer RTL Module
// Implements INT4/INT8 quantized convolution
// 
// Engineer: Sadad Haidari
// Target: XC7Z020CLG400-1 (Zybo Z7-20)
// ============================================================================

`timescale 1ns / 1ps

module conv_layer #(
    parameter INPUT_CHANNELS = 3,
    parameter OUTPUT_CHANNELS = 32,
    parameter KERNEL_SIZE = 3,
    parameter INPUT_WIDTH = 224,
    parameter INPUT_HEIGHT = 224,
    parameter WEIGHT_BITS = 4,
    parameter ACTIVATION_BITS = 8
) (
    input wire clk,
    input wire rst_n,
    
    // Input data interface
    input wire [ACTIVATION_BITS-1:0] input_data,
    input wire input_valid,
    output wire input_ready,
    
    // Weight interface - flattened to avoid array port issues
    input wire [WEIGHT_BITS*256-1:0] weights_flat,  // Flattened weight buffer
    input wire [7:0] weight_addr,
    input wire weight_valid,
    
    // Output data interface
    output wire [ACTIVATION_BITS-1:0] output_data,
    output wire output_valid,
    input wire output_ready,
    
    // Control interface
    input wire start,
    output wire done,
    
    // Configuration - made constant to avoid synthesis issues
    input wire [7:0] stride,
    input wire [7:0] padding,
    input wire [255:0] bias_flat  // 32 bias values * 8 bits each
);

// ============================================================================
// PARAMETERS
// ============================================================================

localparam KERNEL_ELEMENTS = KERNEL_SIZE * KERNEL_SIZE * INPUT_CHANNELS;
localparam OUTPUT_WIDTH = (INPUT_WIDTH + 2 * 1 - KERNEL_SIZE) / 1 + 1;  // Fixed padding/stride
localparam OUTPUT_HEIGHT = (INPUT_HEIGHT + 2 * 1 - KERNEL_SIZE) / 1 + 1;

// ============================================================================
// INTERNAL SIGNALS
// ============================================================================

// State machine
reg [2:0] state;
reg [2:0] next_state;

// Processing counters
reg [9:0] x_counter;
reg [9:0] y_counter;
reg [7:0] out_channel_counter;
reg [7:0] in_channel_counter;
reg [1:0] kernel_x_counter;
reg [1:0] kernel_y_counter;

// Data buffers
reg [ACTIVATION_BITS-1:0] input_buffer [0:KERNEL_SIZE-1][0:KERNEL_SIZE-1];
reg [WEIGHT_BITS-1:0] weight_buffer [0:KERNEL_SIZE-1][0:KERNEL_SIZE-1];
reg [ACTIVATION_BITS-1:0] output_buffer [0:31];

// Accumulation
reg [15:0] accumulator;  // Extended precision for accumulation
reg [15:0] bias_accumulator;

// Control signals
reg processing_active;
reg input_buffer_valid;
reg weight_buffer_valid;

// Weight and bias unpacking
wire [WEIGHT_BITS-1:0] weights [0:255];
wire [7:0] bias [0:31];

// Unpack weights from flattened input
genvar w;
generate
    for (w = 0; w < 256; w = w + 1) begin : WEIGHT_UNPACK
        assign weights[w] = weights_flat[(w+1)*WEIGHT_BITS-1:w*WEIGHT_BITS];
    end
endgenerate

// Unpack bias from flattened input
genvar b;
generate
    for (b = 0; b < 32; b = b + 1) begin : BIAS_UNPACK
        assign bias[b] = bias_flat[(b+1)*8-1:b*8];
    end
endgenerate

// ============================================================================
// STATE MACHINE
// ============================================================================

localparam IDLE = 3'd0;
localparam LOAD_WEIGHTS = 3'd1;
localparam LOAD_INPUT = 3'd2;
localparam CONVOLVE = 3'd3;
localparam ACCUMULATE = 3'd4;
localparam ACTIVATE = 3'd5;
localparam OUTPUT = 3'd6;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
    end else begin
        state <= next_state;
    end
end

always @(*) begin
    next_state = state;
    
    case (state)
        IDLE: begin
            if (start)
                next_state = LOAD_WEIGHTS;
        end
        
        LOAD_WEIGHTS: begin
            if (weight_buffer_valid)
                next_state = LOAD_INPUT;
        end
        
        LOAD_INPUT: begin
            if (input_buffer_valid)
                next_state = CONVOLVE;
        end
        
        CONVOLVE: begin
            if (kernel_x_counter == KERNEL_SIZE-1 && kernel_y_counter == KERNEL_SIZE-1)
                next_state = ACCUMULATE;
        end
        
        ACCUMULATE: begin
            if (in_channel_counter == INPUT_CHANNELS-1)
                next_state = ACTIVATE;
        end
        
        ACTIVATE: begin
            next_state = OUTPUT;
        end
        
        OUTPUT: begin
            if (output_ready) begin
                if (x_counter == OUTPUT_WIDTH-1 && y_counter == OUTPUT_HEIGHT-1 && out_channel_counter == OUTPUT_CHANNELS-1)
                    next_state = IDLE;
                else
                    next_state = LOAD_INPUT;
            end
        end
        
        default: next_state = IDLE;
    endcase
end

// ============================================================================
// COUNTER LOGIC
// ============================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        x_counter <= 10'd0;
        y_counter <= 10'd0;
        out_channel_counter <= 8'd0;
        in_channel_counter <= 8'd0;
        kernel_x_counter <= 2'd0;
        kernel_y_counter <= 2'd0;
    end else begin
        case (state)
            IDLE: begin
                x_counter <= 10'd0;
                y_counter <= 10'd0;
                out_channel_counter <= 8'd0;
                in_channel_counter <= 8'd0;
                kernel_x_counter <= 2'd0;
                kernel_y_counter <= 2'd0;
            end
            
            CONVOLVE: begin
                if (kernel_x_counter == KERNEL_SIZE-1) begin
                    kernel_x_counter <= 2'd0;
                    if (kernel_y_counter == KERNEL_SIZE-1) begin
                        kernel_y_counter <= 2'd0;
                        in_channel_counter <= in_channel_counter + 1;
                    end else begin
                        kernel_y_counter <= kernel_y_counter + 1;
                    end
                end else begin
                    kernel_x_counter <= kernel_x_counter + 1;
                end
            end
            
            OUTPUT: begin
                if (output_ready) begin
                    if (x_counter == OUTPUT_WIDTH-1) begin
                        x_counter <= 10'd0;
                        if (y_counter == OUTPUT_HEIGHT-1) begin
                            y_counter <= 10'd0;
                            if (out_channel_counter == OUTPUT_CHANNELS-1) begin
                                out_channel_counter <= 8'd0;
                            end else begin
                                out_channel_counter <= out_channel_counter + 1;
                            end
                        end else begin
                            y_counter <= y_counter + 1;
                        end
                    end else begin
                        x_counter <= x_counter + 1;
                    end
                end
            end
        endcase
    end
end

// ============================================================================
// INPUT BUFFER LOGIC
// ============================================================================

integer i, j;  // Declare loop variables outside loops

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        input_buffer_valid <= 1'b0;
        for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
            for (j = 0; j < KERNEL_SIZE; j = j + 1) begin
                input_buffer[i][j] <= {ACTIVATION_BITS{1'b0}};
            end
        end
    end else if (state == LOAD_INPUT && input_valid) begin
        // Load input data into buffer with padding
        if (x_counter < INPUT_WIDTH && y_counter < INPUT_HEIGHT) begin
            input_buffer[kernel_x_counter][kernel_y_counter] <= input_data;
        end else begin
            input_buffer[kernel_x_counter][kernel_y_counter] <= {ACTIVATION_BITS{1'b0}};  // Zero padding
        end
        
        if (kernel_x_counter == KERNEL_SIZE-1 && kernel_y_counter == KERNEL_SIZE-1) begin
            input_buffer_valid <= 1'b1;
        end
    end else if (state == CONVOLVE) begin
        input_buffer_valid <= 1'b0;
    end
end

// ============================================================================
// WEIGHT BUFFER LOGIC
// ============================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        weight_buffer_valid <= 1'b0;
        for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
            for (j = 0; j < KERNEL_SIZE; j = j + 1) begin
                weight_buffer[i][j] <= {WEIGHT_BITS{1'b0}};
            end
        end
    end else if (state == LOAD_WEIGHTS && weight_valid) begin
        // Load weights for current output channel
        weight_buffer[kernel_x_counter][kernel_y_counter] <= weights[weight_addr];
        
        if (kernel_x_counter == KERNEL_SIZE-1 && kernel_y_counter == KERNEL_SIZE-1) begin
            weight_buffer_valid <= 1'b1;
        end
    end else if (state == CONVOLVE) begin
        weight_buffer_valid <= 1'b0;
    end
end

// ============================================================================
// CONVOLUTION LOGIC
// ============================================================================

// Multiplier array for parallel convolution
wire [ACTIVATION_BITS+WEIGHT_BITS-1:0] mult_results [0:KERNEL_SIZE-1][0:KERNEL_SIZE-1];

genvar k, l;
generate
    for (k = 0; k < KERNEL_SIZE; k = k + 1) begin : KERNEL_X
        for (l = 0; l < KERNEL_SIZE; l = l + 1) begin : KERNEL_Y
            assign mult_results[k][l] = input_buffer[k][l] * weight_buffer[k][l];
        end
    end
endgenerate

// Accumulation logic
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        accumulator <= 16'h0;
        bias_accumulator <= 16'h0;
    end else if (state == CONVOLVE) begin
        // Accumulate multiplication results
        accumulator <= accumulator + mult_results[kernel_x_counter][kernel_y_counter];
    end else if (state == ACCUMULATE) begin
        // Add bias for current output channel
        bias_accumulator <= accumulator + bias[out_channel_counter];
        accumulator <= 16'h0;
    end else if (state == IDLE) begin
        accumulator <= 16'h0;
        bias_accumulator <= 16'h0;
    end
end

// ============================================================================
// ACTIVATION FUNCTION (ReLU)
// ============================================================================

reg [ACTIVATION_BITS-1:0] activated_output;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        activated_output <= {ACTIVATION_BITS{1'b0}};
    end else if (state == ACTIVATE) begin
        // ReLU activation: max(0, x)
        if (bias_accumulator[15]) begin  // Negative number
            activated_output <= {ACTIVATION_BITS{1'b0}};
        end else begin
            activated_output <= bias_accumulator[ACTIVATION_BITS-1:0];
        end
    end
end

// ============================================================================
// OUTPUT LOGIC
// ============================================================================

assign output_data = activated_output;
assign output_valid = (state == OUTPUT);
assign input_ready = (state == LOAD_INPUT);
assign done = (state == IDLE && processing_active);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        processing_active <= 1'b0;
    end else begin
        if (start)
            processing_active <= 1'b1;
        else if (done)
            processing_active <= 1'b0;
    end
end

endmodule
