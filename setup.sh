#!/bin/bash

# VR Hand Tracking Accelerator Setup Script

echo "Setting up VR Hand Tracking Accelerator environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete! To activate the environment in the future, run:"
echo "source venv/bin/activate"
