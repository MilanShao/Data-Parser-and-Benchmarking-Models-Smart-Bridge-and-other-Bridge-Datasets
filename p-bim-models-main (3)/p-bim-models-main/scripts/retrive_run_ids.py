#!/usr/bin/env python
import argparse
from typing import Any

import wandb

parser = argparse.ArgumentParser()
parser.add_argument(
    "--entity", type=str, default="ma-thesis-kohlmann", help="W&B entity"
)
parser.add_argument("--project", type=str, required=True, help="W&B project")
parser.add_argument("--run-name", type=str, required=False, help="W&B run name")
parser.add_argument("--limit", type=int, required=False)
parser.add_argument("--offset", type=int, required=False)
parser.add_argument("--includes-name", type=str, required=False)

args = parser.parse_args()


def _predicate(run: Any) -> bool:
    if run.state != "finished":
        return False
    if args.run_name:
        return run.name == args.run_name
    if args.includes_name:
        return args.includes_name in run.name
    return True


if __name__ == "__main__":
    api = wandb.Api()
    runs = sorted(
        api.runs(path=f"{args.entity}/{args.project}"), key=lambda run: run.id
    )
    if args.offset:
        runs = runs[args.offset :]
    if args.limit:
        runs = runs[: args.limit]
    ids = [run.id for run in runs if _predicate(run)]
    print(ids)
