import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

import click

from pbim_preprocessor.assembler.grandstand import GrandStandAssembler
from pbim_preprocessor.assembler.lux import LuxAssembler
from pbim_preprocessor.assembler.pbim import PBimAssembler
from pbim_preprocessor.assembler.util import MergeChannelsConfig
from pbim_preprocessor.assembler.wrapper import AssemblerWrapper
from pbim_preprocessor.assembler.z24 import Z24EMSAssembler, Z24PDTAssembler
from pbim_preprocessor.cli.constants import STRATEGIES, FORMATS, CHANNELS, MERGE_CONFIGS
from pbim_preprocessor.index import _write_index
from pbim_preprocessor.metadata import DatasetMetadata, _write_metadata_file
from pbim_preprocessor.model import EOF
from pbim_preprocessor.statistic import StatisticsCollector, ChannelStatistics
from pbim_preprocessor.utils import LOGGER
from pbim_preprocessor.writer import CsvWriter, BinaryWriter

TIME_BYTE_SIZE = 8


def _make_writer(write_type: str, path: Path, headers: List[str], **kwargs):
    match write_type:
        case "csv":
            delimiter = kwargs.get("delimiter", ",")
            return CsvWriter(path, headers, delimiter)
        case "binary":
            return BinaryWriter(path, headers, TIME_BYTE_SIZE)


def _make_metadata(
    mode: str,
    channels: List[str],
    start_time: Optional[datetime.datetime],
    end_time: Optional[datetime.datetime],
    resolution: Optional[int],
    length: int,
    statistics: Dict[str, ChannelStatistics],
    time_byte_size: int,
) -> DatasetMetadata:
    match mode:
        case "pbim":
            return DatasetMetadata(
                channel_order=["Time"] + channels,
                start_time=int(start_time.timestamp()) if start_time else None,
                end_time=int(end_time.timestamp()) if end_time else None,
                # Time + Channels
                measurement_size_in_bytes=time_byte_size + len(channels) * 4,
                resolution=resolution,
                length=length,
                statistics=statistics,
                time_byte_size=time_byte_size,
            )
        case "grandstand":
            return DatasetMetadata(
                channel_order=["Time"] + channels,
                start_time=None,
                end_time=None,
                # Channels (including time)
                measurement_size_in_bytes=(len(channels) - 1) * 4 + time_byte_size,
                resolution=None,
                length=length,
                statistics=statistics,
                time_byte_size=time_byte_size,
            )
        case "z24-ems":
            return DatasetMetadata(
                channel_order=["Time"] + channels,
                start_time=start_time.timestamp() if start_time else None,
                end_time=end_time.timestamp() if end_time else None,
                measurement_size_in_bytes=len(channels) * 4 + time_byte_size,
                resolution=resolution,
                length=length,
                statistics=statistics,
                time_byte_size=time_byte_size,
            )
        case "z24-pdt":
            return DatasetMetadata(
                channel_order=["Time"] + channels,
                start_time=start_time.timestamp() if start_time else None,
                end_time=end_time.timestamp() if end_time else None,
                measurement_size_in_bytes=len(channels) * 4 + time_byte_size,
                resolution=resolution,
                length=length,
                statistics=statistics,
                time_byte_size=time_byte_size,
            )
        case "lux":
            return DatasetMetadata(
                channel_order=["Time"] + channels,
                start_time=start_time.timestamp() if start_time else None,
                end_time=end_time.timestamp() if end_time else None,
                measurement_size_in_bytes=len(channels) * 4 + time_byte_size,
                resolution=resolution,
                length=length,
                statistics=statistics,
                time_byte_size=time_byte_size,
            )


def _validate_args(
    mode: str,
    start_time: Optional[datetime.datetime],
    end_time: Optional[datetime.datetime],
    resolution: Optional[float],
    scenario: Optional[str],
    scenario_type: Optional[str],
):
    def _raise(param: Optional[Any], parameter_name: str):
        if param is None:
            raise click.BadParameter(
                f"Parameter {parameter_name} is required in mode {mode}."
            )

    match mode:
        case "pbim":
            _raise(start_time, "start_time")
            _raise(end_time, "end_time")
            _raise(resolution, "resolution")
        case "grandstand":
            _raise(scenario, "scenario")
        case "z24-ems":
            _raise(resolution, "resolution")
        case "z24-pdt":
            _raise(scenario, "scenario")
            _raise(scenario_type, "scenario_type")
            if scenario_type not in ["avt", "fvt"]:
                raise click.BadParameter(
                    f"Parameter scenario_type must be one of 'avt' or 'fvt'."
                )
        case "lux":
            _raise(start_time, "start_time")
            _raise(end_time, "end_time")
            _raise(resolution, "resolution")


def _prepare_channels(
    mode: str, channels: List[str], scenario_type: Optional[str]
) -> List[str]:
    if not channels:
        return CHANNELS[mode]
    match mode:
        case "pbim":
            if "relevant" in channels:
                ignore_channels = [
                    "MQ_1_MS_U_Neig",
                    "MQ_1_MS_U_Schieb",
                    "MQ_2_MS_U_Neig",
                    "MQ_3_MS_U_Neig",
                    "MQ_4_MS_U_Neig",
                    "MQ_5_MS_U_Neig",
                    "MQ_5_MS_U_Schieb",
                ]
                channels = [c for c in CHANNELS["pbim"] if c not in ignore_channels]
            if "all" in channels:
                channels = CHANNELS["pbim"]
        case "grandstand":
            if "relevant" in channels or "all" in channels:
                channels = CHANNELS["grandstand"]
        case "z24-ems":
            if "all" in channels:
                channels = CHANNELS["z24-ems"]
            if "relevant" in channels:
                channels = [
                    "03",
                    "05",
                    "06",
                    "07",
                    "10",
                    "12",
                    "14",
                    "16",
                    "TBC1",
                    "TBC2",
                ]
        case "z24-pdt":
            if "all" in channels:
                channels = CHANNELS[f"z24-pdt-{scenario_type}"]
        case "lux":
            if "all" in channels:
                channels = CHANNELS["lux"]
    return channels


def _make_assembler(
    mode: str,
    path: Path,
    strategy: str,
    resolution: float,
    temperature_data_path: Optional[Path] = None,
) -> PBimAssembler | GrandStandAssembler | Z24EMSAssembler | Z24PDTAssembler | LuxAssembler:
    match mode:
        case "pbim":
            return PBimAssembler(
                path, STRATEGIES[strategy], resolution, temperature_data_path
            )
        case "grandstand":
            return GrandStandAssembler(path)
        case "z24-ems":
            return Z24EMSAssembler(
                path, STRATEGIES[strategy], resolution, MERGE_CONFIGS["z24-ems"]
            )
        case "z24-pdt":
            return Z24PDTAssembler(path)
        case "lux":
            return LuxAssembler(path, resolution, STRATEGIES[strategy])


def _compute_additional_channels(
    mode: str, temperature_data_available: bool
) -> List[str]:
    match mode:
        case "pbim":
            if temperature_data_available:
                return ["Temperature"]
            else:
                return []

        case _:
            return []


def _compute_actual_channels(
    channels: List[str],
    merge_configs: List[MergeChannelsConfig],
    add_channels: List[str],
):
    actual_channels = channels.copy()
    for merge_config in merge_configs:
        if all(channel in actual_channels for channel in merge_config.channels):
            if merge_config.remove_original:
                for channel in merge_config.channels:
                    actual_channels.remove(channel)
            actual_channels.append(merge_config.name)
        else:
            raise ValueError(
                f"Supposed to merge channels {merge_config.channels} but at least one of them is missing."
            )
    return actual_channels + add_channels


@click.command()
@click.argument(
    "mode", type=click.Choice(["pbim", "grandstand", "z24-ems", "z24-pdt", "lux"])
)
@click.option("--scenario", type=click.STRING)
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.argument(
    "output-path", type=click.Path(writable=True, dir_okay=False, path_type=Path)
)
@click.option("--start-time", default=None, type=click.DateTime())
@click.option("--end-time", default=None, type=click.DateTime())
@click.option("--resolution", default=None, type=click.FLOAT)
@click.option("--strategy", default="mean", type=click.Choice(list(STRATEGIES.keys())))
@click.option("--output-format", default="csv", type=click.Choice(list(FORMATS.keys())))
@click.option("--channel", multiple=True)
@click.option("--scenario-type", default=None, type=click.Choice(["avt", "fvt"]))
@click.option("--is-anomalous", default=False)
@click.option(
    "--temperature-data-path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
)
def assemble(
    mode: str,
    scenario: Optional[str],
    path: Path,
    output_path: Path,
    start_time: Optional[datetime.datetime],
    end_time: Optional[datetime.datetime],
    resolution: Optional[float],
    strategy: Optional[str],
    output_format: str,
    channel: List[str],
    scenario_type: Optional[str],
    is_anomalous: bool,
    temperature_data_path: Optional[Path] = None,
):
    _validate_args(mode, start_time, end_time, resolution, scenario, scenario_type)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    channels_in = _prepare_channels(mode, list(channel), scenario_type)
    additional_channels = _compute_additional_channels(
        mode, temperature_data_path is not None
    )
    channels_out = _compute_actual_channels(
        channels_in, MERGE_CONFIGS[mode], additional_channels
    )

    assembler = AssemblerWrapper(
        mode,
        _make_assembler(mode, path, strategy, resolution, temperature_data_path),
    )

    statistics_collector = StatisticsCollector()
    with _make_writer(output_format, output_path, channels_out) as writer:
        length = 0
        index = []
        for step in assembler.assemble(
            start_time=start_time.replace(tzinfo=datetime.timezone.utc)
            if start_time
            else None,
            end_time=end_time.replace(tzinfo=datetime.timezone.utc)
            if end_time
            else None,
            scenario=scenario,
            channels=channels_in,
            scenario_type=scenario_type,
        ):
            if isinstance(step, dict):
                time = int(step["time"])
                statistics_collector.add_all(step)
                writer.write_step(step, time)
                length += 1
            elif isinstance(step, EOF):
                index += [length]

    _write_metadata_file(
        output_path,
        _make_metadata(
            mode,
            channels_out,
            start_time,
            end_time,
            resolution,
            length,
            statistics_collector.get_all_channel_statistics(),
            TIME_BYTE_SIZE,
        ),
    )
    _write_index([length] if not index else index, is_anomalous, output_path)
