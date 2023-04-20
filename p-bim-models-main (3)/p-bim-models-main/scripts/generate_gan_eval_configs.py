from pathlib import Path

import jinja2

STRATEGIES = ["hourly", "weighted-random", "uniform"]
AGGREGATIONS = ["interpolate", "mean", "nosampling"]
SCENARIOS = ["S1", "S2", "S3"]


class HackUndefined(jinja2.Undefined):
    def __str__(self):
        return f"{{{{{self._undefined_name}}}}}"


def load_template(template_path: Path) -> jinja2.Template:
    loader = jinja2.FileSystemLoader(searchpath=template_path.parent)
    template_environment = jinja2.Environment(loader=loader, undefined=HackUndefined)
    return template_environment.get_template(template_path.name)


def render_template_and_save(template: jinja2.Template, output_path: Path, **kwargs):
    output_path.write_text(template.render(**kwargs))


if __name__ == "__main__":
    config_template = load_template(Path("templates/evaluate_gan_config_template.yml"))
    job_template = load_template(Path("templates/evaluate_gan_job_template.yml"))
    for scenario in SCENARIOS:
        for strategy in STRATEGIES:
            for aggregation in AGGREGATIONS:
                project = f"pbim-reference-gans-{aggregation}-{strategy}"
                output_path_config = Path(
                    f"configs/k8s/evaluation/pbim/gan/{scenario}/{aggregation}-{strategy}.yml"
                )
                output_path_config.parent.mkdir(parents=True, exist_ok=True)
                render_template_and_save(
                    config_template,
                    output_path_config,
                    PROJECT=project,
                    SCENARIO=scenario,
                    AGGREGATION=aggregation,
                    STRATEGY=strategy,
                )
                output_path_job = Path(
                    f"jobs/evaluate/pbim/gan/{scenario}/{aggregation}-{strategy}.yml"
                )
                output_path_job.parent.mkdir(parents=True, exist_ok=True)
                render_template_and_save(
                    job_template,
                    output_path_job,
                    SCENARIO=scenario.lower(),
                    AGGREGATION=aggregation,
                    STRATEGY=strategy,
                    CONFIG_FILE=str(output_path_config),
                )
