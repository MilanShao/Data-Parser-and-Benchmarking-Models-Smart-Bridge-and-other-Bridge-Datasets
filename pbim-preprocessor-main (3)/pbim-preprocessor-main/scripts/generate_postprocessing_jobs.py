#!/usr/bin/env python

from pathlib import Path
import jinja2

EMPIRICAL_SCALING_FACTOR_MEAN_AND_INTERPOLATE = 0.1
EMPIRICAL_SCALING_FACTOR_NOSAMPLING = 0.15

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

SAMPLES_PER_HOUR = 4
SAMPLE_RESOLUTION = {"mean": 25, "interpolate": 25, "nosampling": 75}
DATASET_LENGTH_IN_HOURS = 4 * 24


def load_template(template_path: Path) -> jinja2.Template:
    loader = jinja2.FileSystemLoader(searchpath=template_path.parent)
    template_environment = jinja2.Environment(loader=loader)
    return template_environment.get_template(template_path.name)


def render_template_and_save(template: jinja2.Template, output_path: Path, **kwargs):
    output_path.write_text(template.render(**kwargs))


def get_hourly_strategy_extra_args(
    target_windows: int, window_size: int, dataset_length_in_hours, aggregation: str
) -> str:
    windows_per_sample = target_windows / (SAMPLES_PER_HOUR * dataset_length_in_hours)
    samples_per_hourly_sample = round(windows_per_sample + window_size - 1)
    sample_length = round(samples_per_hourly_sample / SAMPLE_RESOLUTION[aggregation])

    return f"--samples-per-hour {SAMPLES_PER_HOUR} --sample-length {sample_length}"


def get_num_samples(
    target_windows: int, window_size: int, strategy: str, aggregation: str
) -> int:
    # windows = samples - window-size + 1 <=> samples = windows + window-size - 1
    match strategy:
        case "uniform":
            return target_windows + window_size - 1
        case "weighted-random":
            scaling_factor = (
                EMPIRICAL_SCALING_FACTOR_NOSAMPLING
                if aggregation == "nosampling"
                else EMPIRICAL_SCALING_FACTOR_MEAN_AND_INTERPOLATE
            )
            return int(target_windows * window_size * scaling_factor)
        case "hourly":
            raise NotImplementedError("Use get_hourly_strategy_extra_args instead")


def get_extra_args(
    window_size: int, num_windows: int, strategy: str, aggregation: str
) -> str:
    match strategy:
        case "uniform":
            num_samples = get_num_samples(
                num_windows, window_size, strategy, aggregation
            )
            return f"--num-samples {num_samples} --window-size {window_size}"
        case "weighted-random":
            num_samples = get_num_samples(
                num_windows, window_size, strategy, aggregation
            )
            return f"--num-samples {num_samples} --window-size {window_size}"
        case "hourly":
            return get_hourly_strategy_extra_args(
                num_windows, window_size, DATASET_LENGTH_IN_HOURS, aggregation
            )


if __name__ == "__main__":
    NUM_WINDOWS = 100000
    WINDOW_SIZE = 128
    SCENARIO = "N"

    template = load_template(Path("template/postprocess_pbim_job_template.yml"))
    for strategy in ["uniform", "weighted-random", "hourly"]:
        for aggregation in ["nosampling", "mean", "interpolate"]:
            for filename in FILE_NAMES[SCENARIO]:
                input_path_parameter = f"/data/PBIM/{SCENARIO}/assembled/{aggregation}/{filename}/assembled.dat"
                output_path_parameter = f"/data/PBIM/{SCENARIO}/post-processed/{aggregation}-{strategy}/{filename}/assembled.dat"
                output_path = Path(
                    f"k8s/assemble_jobs/pbim/post-process-jobs/{SCENARIO}/{strategy}/{filename}-{aggregation}-{strategy}.yml"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                render_template_and_save(
                    template,
                    output_path,
                    INPUT_FILE=input_path_parameter,
                    OUTPUT_FILE=output_path_parameter,
                    STRATEGY=strategy,
                    STRATEGY_ARGS=get_extra_args(
                        WINDOW_SIZE, NUM_WINDOWS, strategy, aggregation
                    ),
                    AGGREGATION=aggregation,
                    FILENAME=filename,
                    SCENARIO=SCENARIO.lower(),
                    SEED=42,
                )
