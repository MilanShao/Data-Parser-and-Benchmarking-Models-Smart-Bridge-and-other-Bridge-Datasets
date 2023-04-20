#!/bin/bash
set -e
echo "Logging in to Weights & Biases..."
wandb login $WANDB_API_KEY --relogin

echo "Training model..."
python bin/main.py train $CONFIG_FILE