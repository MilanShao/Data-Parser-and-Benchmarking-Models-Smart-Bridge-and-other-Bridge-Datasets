#!/usr/bin/env python3
import argparse

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, required=True, help="W&B project")

args = parser.parse_args()
api = wandb.Api(timeout=15)

# Get all runs in the project that failed/crashed

runs = api.runs(path=f"ma-thesis-kohlmann/{args.project}", filters={"state": {"$in": ["failed", "crashed"]}})
for run in runs:
    run.delete()
