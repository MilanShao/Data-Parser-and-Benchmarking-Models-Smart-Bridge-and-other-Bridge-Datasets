#!/bin/bash
set -e
echo "Logging in to Weights & Biases..."
wandb login $WANDB_API_KEY --relogin

echo "Building config from template for run $RUN_ID"
cat $CONFIG_FILE |\
sed "s/{{RUN_ID}}/$RUN_ID/g" |\
sed "s/{{PROJECT}}/$PROJECT/g" |\
sed "s/{{NAME}}/$NAME/g" > /tmp/config_with_run.yaml

echo "Evaluating model..."
python bin/main.py evaluate /tmp/config_with_run.yaml