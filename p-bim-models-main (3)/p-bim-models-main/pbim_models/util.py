from typing import Tuple

import numpy as np
import torch


def aggregate_errors(
    aggregation: str, errors: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    if isinstance(errors, np.ndarray):
        return _aggregate_errors_np(aggregation, errors)
    elif isinstance(errors, torch.Tensor):
        return _aggregate_errors_torch(aggregation, errors)
    else:
        raise ValueError(f"Unknown type {type(errors)}")


def _aggregate_errors_np(aggregation: str, errors: np.ndarray) -> np.ndarray:
    match aggregation:
        case "mean":
            return np.mean(errors, axis=1)
        case "max":
            return np.max(errors, axis=1)
        case "min":
            return np.min(errors, axis=1)
        case "std":
            return np.std(errors, axis=1)
        case _:
            raise ValueError(f"Unknown aggregation {aggregation}")


def _aggregate_errors_torch(aggregation: str, errors: torch.Tensor) -> torch.Tensor:
    match aggregation:
        case "mean":
            return torch.mean(errors, dim=1)
        case "max":
            return torch.max(errors, dim=1)[0]
        case "min":
            return torch.min(errors, dim=1)[0]
        case "std":
            return torch.std(errors, dim=1)
        case _:
            raise ValueError(f"Unknown aggregation {aggregation}")


def compute_mean_and_covariance(errors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    errors = errors[:, :, 0].reshape(-1, errors.shape[-2])
    mean = np.mean(errors, axis=0)
    cov = np.cov(errors, rowvar=False)
    return mean, cov
