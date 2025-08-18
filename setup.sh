#!/bin/bash

# VR Hand Tracking Accelerator Setup Script
# Author: Sadad Haidari
# Version: 1.0.0

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Header
echo "=================================================================="
echo "  VR Hand Tracking Accelerator - Environment Setup"
echo "=================================================================="
echo ""

# Check Python version
print_status "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Dependencies installed successfully"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Verify installation
print_status "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import onnx; print(f'ONNX version: {onnx.__version__}')"

# Create necessary directories
print_status "Creating project directories..."
mkdir -p data
mkdir -p logs
print_success "Directories created"

# Final instructions
echo ""
echo "=================================================================="
echo "  Setup Complete! 🎉"
echo "=================================================================="
echo ""
print_success "VR Hand Tracking Accelerator environment is ready!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Run the model: python model/cnn.py"
echo "3. Generate annotations: python annotations/generate_annotations.py"
echo ""
echo "For more information, see README.md"
echo "=================================================================="
