#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

import jinja2


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--job-template", type=Path, required=True)
    parser.add_argument("--name", type=str)

    return parser


def load_template(template_path: Path) -> jinja2.Template:
    loader = jinja2.FileSystemLoader(searchpath=template_path.parent)
    template_environment = jinja2.Environment(loader=loader)
    return template_environment.get_template(template_path.name)


def eval_template(template: jinja2.Template, **kwargs):
    return template.render(**kwargs)


def exec_job(job_definition: str):
    command = "kubectl apply -f -"
    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)
    process.communicate(input=job_definition.encode("utf-8"))


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    template = load_template(args.job_template)
    rendered = eval_template(
        template, RUN_ID=args.ids, PROJECT=args.project, NAME=args.name
    )
    exec_job(rendered)
