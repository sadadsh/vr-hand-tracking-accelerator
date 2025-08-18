// ============================================================================
// CNN Accelerator RTL Implementation
// VR Hand Gesture Recognition for Zybo Z7-20
// 
// Engineer: Sadad Haidari
// Target: XC7Z020CLG400-1 (Zybo Z7-20)
// Clock: 200MHz
// 
// Architecture: INT4 weights, INT8 activations
// Model: 319 layers, 89.31% accuracy
// ============================================================================

`timescale 1ns / 1ps

module cnn_accelerator (
    input wire clk,                    // System clock (200MHz)
    input wire rst_n,                  // Active low reset
    
    // AXI-Lite Control Interface
    input wire [31:0] s_axil_awaddr,   // Write address
    input wire s_axil_awvalid,         // Write address valid
    output wire s_axil_awready,        // Write address ready
    input wire [31:0] s_axil_wdata,    // Write data
    input wire [3:0] s_axil_wstrb,     // Write strobes
    input wire s_axil_wvalid,          // Write valid
    output wire s_axil_wready,         // Write ready
    output wire [1:0] s_axil_bresp,    // Write response
    output wire s_axil_bvalid,         // Write response valid
    input wire s_axil_bready,          // Response ready
    input wire [31:0] s_axil_araddr,   // Read address
    input wire s_axil_arvalid,         // Read address valid
    output wire s_axil_arready,        // Read address ready
    output wire [31:0] s_axil_rdata,   // Read data
    output wire [1:0] s_axil_rresp,    // Read response
    output wire s_axil_rvalid,         // Read valid
    input wire s_axil_rready,          // Read ready
    
    // AXI-MM Memory Interface for DDR3
    output wire [31:0] m_axi_awaddr,   // Write address
    output wire [7:0] m_axi_awlen,     // Write burst length
    output wire [2:0] m_axi_awsize,    // Write burst size
    output wire [1:0] m_axi_awburst,   // Write burst type
    output wire m_axi_awvalid,         // Write address valid
    input wire m_axi_awready,          // Write address ready
    output wire [511:0] m_axi_wdata,   // Write data
    output wire [63:0] m_axi_wstrb,    // Write strobes
    output wire m_axi_wlast,           // Write last
    output wire m_axi_wvalid,          // Write valid
    input wire m_axi_wready,           // Write ready
    input wire [1:0] m_axi_bresp,      // Write response
    input wire m_axi_bvalid,           // Write response valid
    output wire m_axi_bready,          // Response ready
    output wire [31:0] m_axi_araddr,   // Read address
    output wire [7:0] m_axi_arlen,     // Read burst length
    output wire [2:0] m_axi_arsize,    // Read burst size
    output wire [1:0] m_axi_arburst,   // Read burst type
    output wire m_axi_arvalid,         // Read address valid
    input wire m_axi_arready,          // Read address ready
    input wire [511:0] m_axi_rdata,    // Read data
    input wire [1:0] m_axi_rresp,      // Read response
    input wire m_axi_rlast,            // Read last
    input wire m_axi_rvalid,           // Read valid
    output wire m_axi_rready           // Read ready
);

// ============================================================================
// PARAMETERS
// ============================================================================

// CNN Architecture Parameters
parameter INPUT_WIDTH = 224;
parameter INPUT_HEIGHT = 224;
parameter INPUT_CHANNELS = 3;
parameter INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;

parameter NUM_CLASSES = 18;
parameter WEIGHT_BITS = 4;      // INT4 weights
parameter ACTIVATION_BITS = 8;  // INT8 activations

// Memory Parameters
parameter BRAM_DEPTH = 1024;
parameter DDR_BURST_LEN = 8;
parameter DDR_BURST_SIZE = 6;   // 64 bytes (512 bits)

// ============================================================================
// INTERNAL SIGNALS
// ============================================================================

// Control registers
reg [31:0] ctrl_reg;
wire [31:0] status_reg;  // Changed to wire for concurrent assignment
reg [31:0] input_addr_reg;
reg [31:0] output_addr_reg;
reg [31:0] weight_addr_reg;

// State machine
reg [3:0] state;
reg [3:0] next_state;

// Processing signals
reg [31:0] input_counter;
reg [31:0] output_counter;
reg [31:0] layer_counter;
reg processing_done;

// Memory interface signals
reg [31:0] mem_addr;
reg [511:0] mem_data_in;
reg [511:0] mem_data_out;
reg mem_read_req;
reg mem_write_req;
reg mem_busy;

// Feature map buffers (BRAM)
reg [ACTIVATION_BITS-1:0] feature_map_a [0:BRAM_DEPTH-1];
reg [ACTIVATION_BITS-1:0] feature_map_b [0:BRAM_DEPTH-1];
reg [9:0] feature_map_addr;
reg feature_map_we_a;
reg feature_map_we_b;
reg [ACTIVATION_BITS-1:0] feature_map_data_in_a;
reg [ACTIVATION_BITS-1:0] feature_map_data_in_b;
wire [ACTIVATION_BITS-1:0] feature_map_data_out_a;
wire [ACTIVATION_BITS-1:0] feature_map_data_out_b;

// Weight buffer (DDR3) - flattened for synthesis
reg [WEIGHT_BITS*256-1:0] weight_buffer_flat;
reg [7:0] weight_addr;
reg weight_we;
reg [WEIGHT_BITS*256-1:0] weight_data_in_flat;

// ============================================================================
// STATE MACHINE DEFINITIONS
// ============================================================================

localparam IDLE = 4'd0;
localparam LOAD_INPUT = 4'd1;
localparam CONV1 = 4'd2;
localparam CONV2 = 4'd3;
localparam CONV3 = 4'd4;
localparam CONV4 = 4'd5;
localparam CONV5 = 4'd6;
localparam FC1 = 4'd7;
localparam FC2 = 4'd8;
localparam SOFTMAX = 4'd9;
localparam STORE_OUTPUT = 4'd10;
localparam DONE = 4'd11;

// ============================================================================
// AXI-LITE SLAVE INTERFACE
// ============================================================================

// AXI-Lite control logic
assign s_axil_awready = 1'b1;
assign s_axil_wready = 1'b1;
assign s_axil_bresp = 2'b00;
assign s_axil_bvalid = 1'b1;
assign s_axil_arready = 1'b1;
assign s_axil_rresp = 2'b00;

// Register read/write logic
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        ctrl_reg <= 32'h0;
        input_addr_reg <= 32'h0;
        output_addr_reg <= 32'h0;
        weight_addr_reg <= 32'h0;
    end else begin
        // Write to control registers
        if (s_axil_awvalid && s_axil_wvalid) begin
            case (s_axil_awaddr[7:0])
                8'h00: ctrl_reg <= s_axil_wdata;
                8'h04: input_addr_reg <= s_axil_wdata;
                8'h08: output_addr_reg <= s_axil_wdata;
                8'h0C: weight_addr_reg <= s_axil_wdata;
            endcase
        end
    end
end

// Read from control registers
assign s_axil_rdata = (s_axil_araddr[7:0] == 8'h00) ? ctrl_reg :
                     (s_axil_araddr[7:0] == 8'h04) ? input_addr_reg :
                     (s_axil_araddr[7:0] == 8'h08) ? output_addr_reg :
                     (s_axil_araddr[7:0] == 8'h0C) ? weight_addr_reg :
                     (s_axil_araddr[7:0] == 8'h10) ? status_reg : 32'h0;

assign s_axil_rvalid = 1'b1;

// ============================================================================
// MAIN STATE MACHINE
// ============================================================================

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
            if (ctrl_reg[0])  // Start bit
                next_state = LOAD_INPUT;
        end
        
        LOAD_INPUT: begin
            if (input_counter >= INPUT_SIZE)
                next_state = CONV1;
        end
        
        CONV1: begin
            if (layer_counter >= 32)  // CONV1 layers
                next_state = CONV2;
        end
        
        CONV2: begin
            if (layer_counter >= 64)  // CONV2 layers
                next_state = CONV3;
        end
        
        CONV3: begin
            if (layer_counter >= 128) // CONV3 layers
                next_state = CONV4;
        end
        
        CONV4: begin
            if (layer_counter >= 256) // CONV4 layers
                next_state = CONV5;
        end
        
        CONV5: begin
            if (layer_counter >= 288) // CONV5 layers
                next_state = FC1;
        end
        
        FC1: begin
            if (layer_counter >= 304) // FC1 layers
                next_state = FC2;
        end
        
        FC2: begin
            if (layer_counter >= 312) // FC2 layers
                next_state = SOFTMAX;
        end
        
        SOFTMAX: begin
            if (layer_counter >= 319) // All layers done
                next_state = STORE_OUTPUT;
        end
        
        STORE_OUTPUT: begin
            if (output_counter >= NUM_CLASSES)
                next_state = DONE;
        end
        
        DONE: begin
            next_state = IDLE;
        end
        
        default: next_state = IDLE;
    endcase
end

// ============================================================================
// PROCESSING LOGIC
// ============================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        input_counter <= 32'h0;
        output_counter <= 32'h0;
        layer_counter <= 32'h0;
        processing_done <= 1'b0;
    end else begin
        case (state)
            LOAD_INPUT: begin
                if (!mem_busy) begin
                    input_counter <= input_counter + 1;
                end
            end
            
            CONV1, CONV2, CONV3, CONV4, CONV5: begin
                if (!mem_busy) begin
                    layer_counter <= layer_counter + 1;
                end
            end
            
            FC1, FC2: begin
                if (!mem_busy) begin
                    layer_counter <= layer_counter + 1;
                end
            end
            
            SOFTMAX: begin
                if (!mem_busy) begin
                    layer_counter <= layer_counter + 1;
                end
            end
            
            STORE_OUTPUT: begin
                if (!mem_busy) begin
                    output_counter <= output_counter + 1;
                end
            end
            
            DONE: begin
                processing_done <= 1'b1;
            end
            
            default: begin
                if (state == IDLE) begin
                    input_counter <= 32'h0;
                    output_counter <= 32'h0;
                    layer_counter <= 32'h0;
                    processing_done <= 1'b0;
                end
            end
        endcase
    end
end

// Status register - concurrent assignment
assign status_reg = {28'h0, processing_done, state};

// ============================================================================
// MEMORY INTERFACE
// ============================================================================

// AXI-MM master interface for DDR3
assign m_axi_awaddr = mem_addr;
assign m_axi_awlen = DDR_BURST_LEN - 1;
assign m_axi_awsize = DDR_BURST_SIZE;
assign m_axi_awburst = 2'b01;  // INCR
assign m_axi_awvalid = mem_read_req;
assign m_axi_wdata = mem_data_in;
assign m_axi_wstrb = 64'hFFFFFFFFFFFFFFFF;
assign m_axi_wlast = 1'b1;
assign m_axi_wvalid = mem_write_req;
assign m_axi_bready = 1'b1;
assign m_axi_araddr = mem_addr;
assign m_axi_arlen = DDR_BURST_LEN - 1;
assign m_axi_arsize = DDR_BURST_SIZE;
assign m_axi_arburst = 2'b01;  // INCR
assign m_axi_arvalid = mem_read_req;
assign m_axi_rready = 1'b1;

// Memory control logic
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        mem_busy <= 1'b0;
        mem_addr <= 32'h0;
        mem_data_in <= 512'h0;
    end else begin
        case (state)
            LOAD_INPUT: begin
                if (!mem_busy && m_axi_arready) begin
                    mem_addr <= input_addr_reg + (input_counter << 2);
                    mem_busy <= 1'b1;
                end else if (m_axi_rvalid) begin
                    mem_data_in <= m_axi_rdata;
                    mem_busy <= 1'b0;
                end
            end
            
            STORE_OUTPUT: begin
                if (!mem_busy && m_axi_awready) begin
                    mem_addr <= output_addr_reg + (output_counter << 2);
                    mem_busy <= 1'b1;
                end else if (m_axi_wready) begin
                    mem_busy <= 1'b0;
                end
            end
        endcase
    end
end

// ============================================================================
// FEATURE MAP BRAM INSTANTIATION
// ============================================================================

// Feature map A (input/current layer)
always @(posedge clk) begin
    if (feature_map_we_a) begin
        feature_map_a[feature_map_addr] <= feature_map_data_in_a;
    end
end
assign feature_map_data_out_a = feature_map_a[feature_map_addr];

// Feature map B (output/next layer)
always @(posedge clk) begin
    if (feature_map_we_b) begin
        feature_map_b[feature_map_addr] <= feature_map_data_in_b;
    end
end
assign feature_map_data_out_b = feature_map_b[feature_map_addr];

// ============================================================================
// WEIGHT BUFFER
// ============================================================================

always @(posedge clk) begin
    if (weight_we) begin
        weight_buffer_flat <= weight_data_in_flat;
    end
end

// ============================================================================
// CNN LAYER INSTANTIATIONS
// ============================================================================

// Example: CONV1 layer
wire [ACTIVATION_BITS-1:0] conv1_output;
wire conv1_done;

conv_layer #(
    .INPUT_CHANNELS(3),
    .OUTPUT_CHANNELS(32),
    .KERNEL_SIZE(3),
    .INPUT_WIDTH(224),
    .INPUT_HEIGHT(224),
    .WEIGHT_BITS(WEIGHT_BITS),
    .ACTIVATION_BITS(ACTIVATION_BITS)
) conv1_inst (
    .clk(clk),
    .rst_n(rst_n),
    .input_data(feature_map_data_out_a),
    .input_valid(1'b1),
    .input_ready(),
    .weights_flat(weight_buffer_flat),
    .weight_addr(weight_addr),
    .weight_valid(weight_we),
    .output_data(conv1_output),
    .output_valid(conv1_done),
    .output_ready(1'b1),
    .start(state == CONV1),
    .done(conv1_done),
    .stride(8'd1),
    .padding(8'd1),
    .bias_flat(256'h0)  // Placeholder bias - 32 values * 8 bits each
);

// Additional layers would be instantiated here...

endmodule
