import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from dataclasses_json import dataclass_json, config
from marshmallow import fields


def _path_field(default: Optional[Path] = None):
    return field(
        metadata=config(
            encoder=str,
            decoder=lambda x: Path(x) if x is not None else default,
            mm_field=fields.String(),
        )
    )


@dataclass_json
@dataclass
class FileConfig:
    is_anomalous: bool
    relative_path: Path = _path_field()
    offset: float = 0.0
    ratio: float = 1.0
    include_in_statistics: bool = True


@dataclass_json
@dataclass
class MergeConfig:
    base_path: Path = _path_field()
    output_file: Path = _path_field()
    use_statistics_from: Optional[Path] = None
    start: Optional[datetime.datetime] = None
    end: Optional[datetime.datetime] = None
    files: Optional[List[FileConfig]] = None
    keep_statistics: bool = False
