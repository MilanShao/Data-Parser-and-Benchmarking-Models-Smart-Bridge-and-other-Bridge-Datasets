from dataclasses import dataclass
from typing import List


@dataclass
class MergeChannelsConfig:
    channels: List[str]
    name: str
    remove_original: bool = True
