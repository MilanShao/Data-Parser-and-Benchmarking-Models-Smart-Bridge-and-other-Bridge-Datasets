#!/bin/bash
set -e
echo "Logging in to Weights & Biases..."
wandb login

echo "Starting sweep..."
echo "Sweep ID: $SWEEP_ID"
if [ -z $COUNT ]; then
  wandb agent $SWEEP_ID
else
  echo "Limiting number of runs to $COUNT"
  wandb agent $SWEEP_ID --count $COUNT
fi

