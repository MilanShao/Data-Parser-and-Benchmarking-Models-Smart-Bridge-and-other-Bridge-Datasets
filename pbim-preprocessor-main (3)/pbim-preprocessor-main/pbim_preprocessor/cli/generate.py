from dataclasses import dataclass, field
import datetime
from pathlib import Path
from typing import List

import click
import jinja2
from dataclasses_json import dataclass_json, config
from marshmallow import fields


def date_field():
    return field(
        metadata=config(
            encoder=lambda x: x.strftime("%Y-%m-%dT%H:%M:%S"),
            decoder=lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S"),
            mm_field=fields.DateTime(format="%Y-%m-%dT%H:%M:%S"),
        )
    )


def path_field():
    return field(
        metadata=config(
            encoder=lambda x: str(x), decoder=Path, mm_field=fields.String()
        )
    )


@dataclass_json
@dataclass
class AssemblyConfig:
    identifier: str
    meta: field(default_factory=dict)

    def __getattr__(self, item):
        if item in self.meta:
            return self.meta[item]
        else:
            return super().__getattr__(item)

    def to_flat_dict(self):
        return {"identifier": self.identifier, **self.meta}


@dataclass_json
@dataclass
class Config:
    assembly_configs: list[AssemblyConfig]
    template_path: Path = path_field()
    output_path: Path = path_field()


def load_template(template_path: Path) -> jinja2.Template:
    loader = jinja2.FileSystemLoader(searchpath=template_path.parent)
    template_environment = jinja2.Environment(loader=loader)
    return template_environment.get_template(template_path.name)


@click.command()
@click.argument(
    "config-path", type=click.Path(exists=True, file_okay=True, path_type=Path)
)
def generate(config_path: Path):
    config = Config.from_json(config_path.read_text())
    template = load_template(config.template_path)
    config.output_path.mkdir(parents=True, exist_ok=True)
    for assembly_config in config.assembly_configs:
        output_path = config.output_path / f"{assembly_config.identifier}.yml"
        output_path.write_text(template.render(assembly_config.to_flat_dict()))
