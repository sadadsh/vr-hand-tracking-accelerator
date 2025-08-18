# Makefile for CNN Accelerator HLS Project

PROJECT = cnn_accelerator_hls
SOLUTION = solution1
TOP_FUNCTION = cnn_accelerator

.PHONY: all clean synthesis cosim export

all: synthesis

synthesis:
	@echo "Running HLS synthesis..."
	vivado_hls -f build_hls.tcl

cosim: synthesis
	@echo "Running co-simulation..."
	# Add co-simulation commands here

export: synthesis
	@echo "Exporting IP..."
	# IP export is included in build_hls.tcl

clean:
	@echo "Cleaning project..."
	rm -rf $(PROJECT)
	rm -rf *.log

help:
	@echo "Available targets:"
	@echo "  synthesis - Run HLS synthesis"
	@echo "  cosim     - Run co-simulation"
	@echo "  export    - Export IP"
	@echo "  clean     - Clean project"
