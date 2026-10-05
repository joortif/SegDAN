

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence
import torch


class BaseMetricTracker(ABC):
    registry: Dict[str, dict] = {}

    def __init__(
        self,
        metric_names: Sequence[str],
        stages: Sequence[str] = ("valid", "test"),
        worst_score: float = 30.0,
        class_thresholds: Sequence[float] = (1.0,),
        every_n_epochs: int = 1,
        skip_empty_gt: bool = True,
    ):
        self.names = [n for n in map(str, metric_names) if n in self.registry]
        self.stages = set(stages)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.skip_empty_gt = skip_empty_gt
        self.cfg = {"worst_score": worst_score, "class_thresholds": list(class_thresholds)}
 
        self._state: Dict[str, Any] = {}
        self._seen: Dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return len(self.names) > 0
     
    def is_active(self, stage: str, epoch: Optional[int] = None) -> bool:
        if stage not in self.stages:
            return False
        if epoch is not None and stage == "valid" and epoch % self.every_n_epochs:
            return False
        return True

    def check_binary(self, binary: bool) -> None:
        if self.enabled and not binary:
            raise ValueError(
                f"{type(self).__name__} only supports binary segmentation."
            )

    def reset(self, stage: str) -> None:
            if not self.is_active(stage):
                return
            self._scores[stage] = {name: [] for name in self.names}
            self._seen[stage] = 0

    @torch.no_grad()
    def update(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor, stage: str) -> None:
        if not self.enabled or stage not in self.stages:
            return
        if stage not in self._state:
            self.reset(stage)
 
        pred = pred_mask.detach().float().cpu()
        gt = gt_mask.detach().float().cpu()
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if gt.ndim == 3:
            gt = gt.unsqueeze(1)
        if pred.ndim != 4 or pred.shape[1] != 1 or pred.shape != gt.shape:
            raise ValueError(
                f"{type(self).__name__} only supports binary 2D segmentation: expected (B, 1, H, W) "
                f"o (B, H, W) for both masks, got {tuple(pred_mask.shape)} / "
                f"{tuple(gt_mask.shape)}"
            )
 
        if self.skip_empty_gt:
            keep = gt.flatten(1).sum(1) > 0
            if not keep.any():
                return
            pred, gt = pred[keep], gt[keep]
 
        self._update_state(self._state[stage], pred, gt)
        self._seen[stage] += pred.shape[0]
 
    def compute(self, stage: str) -> Dict[str, float]:
        if not self.enabled or self._seen.get(stage, 0) == 0:
            return {}
        return self._compute_state(self._state[stage])
 
    @abstractmethod
    def _new_state(self) -> Any:
        ...
 
    @abstractmethod
    def _update_state(self, state: Any, pred: torch.Tensor, gt: torch.Tensor) -> None:
        ...
 
    @abstractmethod
    def _compute_state(self, state: Any) -> Dict[str, float]:
        ...