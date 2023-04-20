from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from torch.utils.data import DataLoader

from pbim_models.dataset.dataset import WindowedDataset, _merge
from pbim_models.dataset.transform import Transforms


class CollateWrapper:
    @staticmethod
    def _collate(data: List[Dict[str, Any]]) -> Tuple[Dict[str, Any]]:
        return _merge(data)

    def __call__(self, data: List[Dict[str, Any]]) -> Tuple[Dict[str, Any]]:
        return self._collate(data)


def build_dataloader(
    path: Path,
    batch_size: int,
    window_size: int,
    horizon_size: int,
    offset=0,
    limit: Optional[int] = None,
    workers=0,
    shuffle=True,
    base_transform=Transforms.identity_transform,
    transform=Transforms.identity_transform,
    lazy_init=False,
    drop_last=True,
    mirror=False,
    in_memory=False,
):
    dataset = WindowedDataset(
        path,
        window_size=window_size,
        horizon_size=horizon_size,
        offset=offset,
        limit=limit,
        base_transform=base_transform,
        transform=transform,
        lazy_init=lazy_init,
        mirror=mirror,
        in_memory=in_memory,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=shuffle,
        collate_fn=CollateWrapper(),
        drop_last=drop_last,
    )
