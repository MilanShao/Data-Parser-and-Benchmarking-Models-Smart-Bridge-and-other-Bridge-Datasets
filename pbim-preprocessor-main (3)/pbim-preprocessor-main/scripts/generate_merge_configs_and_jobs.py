#!/usr/bin/env python
import json
from pathlib import Path
import jinja2

FILE_NAMES = {
    # Normal
    "N": [
        "april-week-01",
        "january-week-01",
        "june-week-01",
        "may-week-01",
        "february-week-01",
        "july-week-01",
        "march-week-01",
    ],
    # Damaged
    "S1": [
        "july-week-02",
        "august-week-01",
    ],
    "S2": [
        "july-week-02",
        "august-week-01",
    ],
    "S3": [
        "july-week-02",
        "august-week-01",
    ],
}


def get_files(scenario: str, strategy: str, aggregation: str):
    def block(s: str, name: str, anomalous: bool):
        return {
            "relative_path": f"{s}/post-processed/{aggregation}-{strategy}/{name}/assembled.dat",
            "is_anomalous": anomalous,
            "include_in_statistics": scenario == "N",
        }

    match scenario:
        case "N":
            return [
                block("N", "january-week-01", False),
                block("N", "april-week-01", False),
                block("N", "july-week-01", False),
            ]
        case _:
            return [
                block("N", "february-week-01", False),
                block("N", "june-week-01", False),
                block(scenario, "july-week-02", True),
                block(scenario, "august-week-01", True),
            ]


def load_template(template_path: Path) -> jinja2.Template:
    loader = jinja2.FileSystemLoader(searchpath=template_path.parent)
    template_environment = jinja2.Environment(loader=loader)
    return template_environment.get_template(template_path.name)


def render_template_and_save(template: jinja2.Template, output_path: Path, **kwargs):
    output_path.write_text(template.render(**kwargs))


def get_output_path_config(scenario: str, strategy: str, aggregation: str):
    match scenario:
        case "N":
            return Path(f"configs/pbim/{scenario}/merge-{aggregation}-{strategy}.json")
        case _:
            return Path(
                f"configs/pbim/anomalous/{scenario}/merge-{aggregation}-{strategy}.json"
            )


def get_dataset_output_path(scenario: str, strategy: str, aggregation: str):
    match scenario:
        case "N":
            return f"/data/PBIM/{scenario}/merged/reference/{aggregation}-{strategy}/assembled.dat"
        case _:
            return f"/data/PBIM/{scenario}/merged/anomalous/reference/{aggregation}-{strategy}/assembled.dat"


if __name__ == "__main__":
    job_template = load_template(Path("template/merge_pbim_job_template.yml"))
    merge_config_template = load_template(
        Path("template/merge_pbim_config_template.json")
    )
    for scenario in ["N", "S1", "S2", "S3"]:
        for strategy in ["hourly", "uniform", "weighted-random"]:
            for aggregation in ["nosampling", "mean", "interpolate"]:
                output_path_job = Path(
                    f"k8s/assemble_jobs/pbim/post-process-jobs/{scenario}/merge/merge-{aggregation}-{strategy}.yml"
                )
                output_path_config = get_output_path_config(
                    scenario, strategy, aggregation
                )
                config_path_parameter = Path("/app") / output_path_config
                output_path_config.parent.mkdir(parents=True, exist_ok=True)
                output_path_job.parent.mkdir(parents=True, exist_ok=True)
                render_template_and_save(
                    merge_config_template,
                    output_path_config,
                    STRATEGY=strategy,
                    OUTPUT_FILE=get_dataset_output_path(
                        scenario, strategy, aggregation
                    ),
                    AGGREGATION=aggregation,
                    SCENARIO=scenario,
                    FILES=json.dumps(
                        get_files(scenario, strategy, aggregation), indent=4
                    ),
                )
                render_template_and_save(
                    job_template,
                    output_path_job,
                    STRATEGY=strategy,
                    AGGREGATION=aggregation,
                    SCENARIO=scenario.lower(),
                    CONFIG_PATH=config_path_parameter,
                )
