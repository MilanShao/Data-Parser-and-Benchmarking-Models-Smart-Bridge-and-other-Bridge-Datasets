import struct
from pathlib import Path

import click

from pbim_preprocessor.utils import _load_metadata


@click.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.argument("output-file", type=click.Path(writable=True, path_type=Path))
@click.argument("max-gap", type=click.INT)
def index(data: Path, output_file: Path, max_gap: int):
    metadata = _load_metadata(data)
    chunk_size = metadata.measurement_size_in_bytes * 1000
    with open(data, "rb") as f, open(output_file, "w") as o:
        current_time = struct.unpack("<i", f.read(4))[0]
        f.seek(0)
        window_start_byte = 0
        num_chunks = 0
        while True:
            chunk = f.read(chunk_size)
            if len(chunk) == 0:
                break
            for i in range(0, len(chunk), metadata.measurement_size_in_bytes):
                timestamp = struct.unpack("<i", chunk[i : i + 4])[0]
                if timestamp - current_time > max_gap:
                    window_end_byte = num_chunks * chunk_size + i
                    o.write(f"{window_start_byte},{window_end_byte}")
                window_start_byte = (
                    num_chunks * chunk_size + i + metadata.measurement_size_in_bytes
                )
                current_time = timestamp
            num_chunks += 1
