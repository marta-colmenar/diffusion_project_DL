#!/bin/bash

# Activate the virtual environment if needed
# source venv/bin/activate

# Set the Python path to the src directory
export PYTHONPATH=$(pwd)/src

# Run the training script
python src/train.py --config configs/train.yaml