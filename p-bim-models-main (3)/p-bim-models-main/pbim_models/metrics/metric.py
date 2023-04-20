import abc
from typing import List

import torch
import torchmetrics
from torchmetrics import Metric


class TimeSeriesMetric(Metric):
    @abc.abstractmethod
    def is_classification_metric(self) -> bool:
        pass

    @abc.abstractmethod
    def returns_dict(self) -> bool:
        pass


class TimeSeriesLPMetric(TimeSeriesMetric):
    def __init__(self, p: float, root: bool = False, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.p = p
        self.root = root
        self.add_state("_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape
        error = (torch.abs(preds - targets) ** self.p).sum(dim=-2)
        if self.reduction == "mean":
            error = error.mean()
        elif self.reduction == "sum":
            error = error.sum()
        else:
            raise ValueError(f"Unknown reduction with '{self.reduction}'.")
        self._error += error
        self._count += 1

    def compute(self):
        if self.root:
            return (self._error / self._count) ** (1 / self.p)
        else:
            return self._error / self._count

    def is_classification_metric(self) -> bool:
        return False

    def returns_dict(self) -> bool:
        return False


class TimeSeriesMAE(TimeSeriesLPMetric):
    def __init__(self, reduction: str = "mean"):
        super().__init__(p=1, reduction=reduction)


class TimeSeriesMSE(TimeSeriesLPMetric):
    def __init__(self, reduction: str = "mean"):
        super().__init__(p=2, reduction=reduction)


class TimeSeriesRMSE(TimeSeriesLPMetric):
    def __init__(self, reduction: str = "mean"):
        super().__init__(p=2, reduction=reduction, root=True)


class TimeSeriesMAPE(TimeSeriesMetric):
    def __init__(self):
        super().__init__()
        self.add_state("_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape
        relative_error = torch.linalg.norm(
            torch.abs((preds - targets) / targets), ord=2, dim=-2
        )
        error = relative_error.mean(dim=1).mean()
        self._error += error
        self._count += preds.shape[0]

    def compute(self):
        return self._error / self._count * 100

    def is_classification_metric(self) -> bool:
        return False

    def returns_dict(self) -> bool:
        return False


class ErrorDistributionMetric(TimeSeriesMetric):
    def __init__(self):
        super().__init__()
        self.add_state("_errors_normal", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state(
            "_errors_normal_squared", default=torch.zeros(1), dist_reduce_fx="sum"
        )
        self.add_state("_counts_normal", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state(
            "_errors_anomalous", default=torch.zeros(1), dist_reduce_fx="sum"
        )
        self.add_state(
            "_errors_anomalous_squared", default=torch.zeros(1), dist_reduce_fx="sum"
        )
        self.add_state(
            "_counts_anomalous", default=torch.zeros(1), dist_reduce_fx="sum"
        )

    def update(self, errors: torch.Tensor, mask: torch.Tensor):
        mask = mask.to(dtype=torch.long)
        self._errors_normal += errors[mask == 0].sum()
        self._errors_normal_squared += torch.sum(errors[mask == 0] ** 2)
        self._counts_normal += torch.count_nonzero(mask == 0)
        self._errors_anomalous += errors[mask == 1].sum()
        self._errors_anomalous_squared += torch.sum(errors[mask == 1] ** 2)
        self._counts_anomalous += torch.count_nonzero(mask)

    def compute(self):
        return {
            "normal": self._errors_normal / self._counts_normal,
            "normal_squared": self._errors_normal_squared / self._counts_normal,
            "anomalous": self._errors_anomalous / self._counts_anomalous,
            "anomalous_squared": self._errors_anomalous_squared
            / self._counts_anomalous,
        }

    def is_classification_metric(self) -> bool:
        return True

    def returns_dict(self) -> bool:
        return True


def _labels(probs: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
    if len(probs.shape) == 1:
        repeated_probs = probs.unsqueeze(1)
    else:
        repeated_probs = probs
    repeated_probs = repeated_probs.repeat(1, ks.shape[0])
    repeated_ks = ks.unsqueeze(0).repeat(probs.shape[0], 1)
    assert repeated_probs.shape == repeated_ks.shape
    return torch.where(
        repeated_probs > repeated_ks,
        torch.ones_like(repeated_probs),
        torch.zeros_like(repeated_probs),
    )


def tp(
    probs: torch.Tensor, mask: torch.Tensor, ks: torch.Tensor, reduce: bool = True
) -> torch.Tensor:
    repeated_mask = mask.unsqueeze(1).repeat(1, ks.shape[0]).to(dtype=torch.long)
    labels = _labels(probs, ks)
    tp = torch.where(labels == 1, repeated_mask, torch.zeros_like(repeated_mask))
    return tp.sum(dim=0) if reduce else tp


def fp(
    probs: torch.Tensor, mask: torch.Tensor, ks: torch.Tensor, reduce: bool = True
) -> torch.Tensor:
    repeated_mask = mask.unsqueeze(1).repeat(1, ks.shape[0]).to(dtype=torch.long)
    labels = _labels(probs, ks)
    fp = torch.where(labels == 1, 1 - repeated_mask, torch.zeros_like(repeated_mask))
    return fp.sum(dim=0) if reduce else fp


def tn(
    probs: torch.Tensor, mask: torch.Tensor, ks: torch.Tensor, reduce: bool = True
) -> torch.Tensor:
    repeated_mask = mask.unsqueeze(1).repeat(1, ks.shape[0]).to(dtype=torch.long)
    labels = _labels(probs, ks)
    tn = torch.where(labels == 0, 1 - repeated_mask, torch.zeros_like(repeated_mask))
    return tn.sum(dim=0) if reduce else tn


def fn(
    probs: torch.Tensor, mask: torch.Tensor, ks: torch.Tensor, reduce: bool = True
) -> torch.Tensor:
    repeated_mask = mask.unsqueeze(1).repeat(1, ks.shape[0]).to(dtype=torch.long)
    labels = _labels(probs, ks)
    fn = torch.where(labels == 0, repeated_mask, torch.zeros_like(repeated_mask))
    return fn.sum(dim=0) if reduce else fn


class AUROC(TimeSeriesMetric):
    def __init__(self, ks: List[float]):
        super().__init__()
        self.add_state("_ks", default=torch.tensor(ks), dist_reduce_fx=None)
        self.add_state("_ks", default=torch.tensor(ks), dist_reduce_fx="sum")
        self.add_state("_tp", default=torch.zeros(len(ks)), dist_reduce_fx="sum")
        self.add_state("_fp", default=torch.zeros(len(ks)), dist_reduce_fx="sum")
        self.add_state("_tn", default=torch.zeros(len(ks)), dist_reduce_fx="sum")
        self.add_state("_fn", default=torch.zeros(len(ks)), dist_reduce_fx="sum")

    def update(self, probs: torch.Tensor, mask: torch.Tensor):
        self._tp += tp(probs, mask, self._ks)
        self._fp += fp(probs, mask, self._ks)
        self._tn += tn(probs, mask, self._ks)
        self._fn += fn(probs, mask, self._ks)

    def compute(self):
        total_positive = self._tp + self._fn
        total_negative = self._fp + self._tn
        tpr = torch.where(
            total_positive > 0, self._tp / total_positive, torch.zeros_like(self._tp)
        )
        fpr = torch.where(
            total_negative > 0, self._fp / total_negative, torch.zeros_like(self._fp)
        )
        return torch.trapz(tpr.flip(dims=[0]), fpr.flip(dims=[0]))

    def is_classification_metric(self) -> bool:
        return True

    def returns_dict(self) -> bool:
        return False


class Precision(TimeSeriesMetric):
    def __init__(self, ks: List[float]):
        super().__init__()
        self.add_state("_ks", default=torch.tensor(ks), dist_reduce_fx=None)
        self.add_state("_tp", default=torch.zeros(len(ks)), dist_reduce_fx="sum")
        self.add_state("_fp", default=torch.zeros(len(ks)), dist_reduce_fx="sum")

    def update(self, probs: torch.Tensor, mask: torch.Tensor):
        self._tp += tp(probs, mask, self._ks)
        self._fp += fp(probs, mask, self._ks)

    def compute(self):
        total_positive = self._tp + self._fp
        precision = torch.where(
            total_positive > 0, self._tp / total_positive, torch.ones_like(self._tp)
        )
        return {f"precision@{k:.4f}": p for k, p in zip(self._ks, precision)}

    def is_classification_metric(self) -> bool:
        return True

    def returns_dict(self) -> bool:
        return True


class Recall(TimeSeriesMetric):
    def __init__(self, ks: List[float]):
        super().__init__()
        self.add_state("_ks", default=torch.tensor(ks), dist_reduce_fx=None)
        self.add_state("_tp", default=torch.zeros(len(ks)), dist_reduce_fx="sum")
        self.add_state("_fn", default=torch.zeros(len(ks)), dist_reduce_fx="sum")

    def update(self, probs: torch.Tensor, mask: torch.Tensor):
        self._tp += tp(probs, mask, self._ks)
        self._fn += fn(probs, mask, self._ks)

    def compute(self):
        total_positive = self._tp + self._fn
        recall = torch.where(
            total_positive > 0, self._tp / total_positive, torch.ones_like(self._tp)
        )
        return {f"recall@{k:.4f}": r for k, r in zip(self._ks, recall)}

    def is_classification_metric(self) -> bool:
        return True

    def returns_dict(self) -> bool:
        return True


class F1(TimeSeriesMetric):
    def __init__(self, ks: List[float]):
        super().__init__()
        self.add_state("_ks", default=torch.tensor(ks), dist_reduce_fx=None)
        self.add_state("_tp", default=torch.zeros(len(ks)), dist_reduce_fx="sum")
        self.add_state("_fp", default=torch.zeros(len(ks)), dist_reduce_fx="sum")
        self.add_state("_fn", default=torch.zeros(len(ks)), dist_reduce_fx="sum")

    def update(self, probs: torch.Tensor, mask: torch.Tensor):
        self._tp += tp(probs, mask, self._ks)
        self._fp += fp(probs, mask, self._ks)
        self._fn += fn(probs, mask, self._ks)

    def compute(self):
        total_positive = self._tp + self._fp
        precision = torch.where(
            total_positive > 0, self._tp / total_positive, torch.ones_like(self._tp)
        )
        actual_positive = self._tp + self._fn
        recall = torch.where(
            actual_positive > 0, self._tp / actual_positive, torch.ones_like(self._tp)
        )
        f1 = 2 * (precision * recall) / (precision + recall)
        return {f"f1@{k:.4f}": f for k, f in zip(self._ks, f1)}

    def is_classification_metric(self) -> bool:
        return True

    def returns_dict(self) -> bool:
        return True


class Accuracy(TimeSeriesMetric):
    def __init__(self, ks: List[float]):
        super().__init__()
        self.add_state("_ks", default=torch.tensor(ks), dist_reduce_fx=None)
        self.add_state("_tp", default=torch.zeros(len(ks)), dist_reduce_fx="sum")
        self.add_state("_tn", default=torch.zeros(len(ks)), dist_reduce_fx="sum")
        self.add_state("_count", default=torch.zeros(len(ks)), dist_reduce_fx="sum")

    def update(self, probs: torch.Tensor, mask: torch.Tensor):
        self._tp += tp(probs, mask, self._ks)
        self._tn += tn(probs, mask, self._ks)
        self._count += mask.shape[0]

    def compute(self):
        accuracy = (self._tp + self._tn) / self._count
        return {f"accuracy@{k:.4f}": a for k, a in zip(self._ks, accuracy)}

    def is_classification_metric(self) -> bool:
        return True

    def returns_dict(self) -> bool:
        return True
