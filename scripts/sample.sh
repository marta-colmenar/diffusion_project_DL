#!/bin/bash

# Activate the virtual environment if needed
# source venv/bin/activate

# Set the necessary environment variables
export PYTHONPATH=$(pwd)/src

# Run the sampling script
python -m src.sample