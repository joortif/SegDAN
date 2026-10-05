import warnings
from typing import Dict, Sequence
 
import torch
from CCMetrics import CCDiceMetric, CCHausdorffDistanceMetric, CCHausdorffDistance95Metric, CCSurfaceDistanceMetric, CCSurfaceDiceMetric 

from segdan.metrics.basetracker import BaseMetricTracker
 
CC_METRICS_REGISTRY = {
    "dice_cc": {
        "builder": lambda cfg: CCDiceMetric(
            cc_reduction=cfg["cc_reduction"],
            use_caching=False,
        ),
        "higher_is_better": True,
    },
    "hd_cc": {
        "builder": lambda cfg: CCHausdorffDistanceMetric(
            cc_reduction=cfg["cc_reduction"],
            use_caching=False,
            metric_worst_score=cfg["worst_score"],
        ),
        "higher_is_better": False,
    },
    "hd95_cc": {
        "builder": lambda cfg: CCHausdorffDistance95Metric(
            cc_reduction=cfg["cc_reduction"],
            use_caching=False,
            metric_worst_score=cfg["worst_score"],
        ),
        "higher_is_better": False,
    },
    "assd_cc": {
        "builder": lambda cfg: CCSurfaceDistanceMetric(
            cc_reduction=cfg["cc_reduction"],
            use_caching=False,
            metric_worst_score=cfg["worst_score"],
        ),
        "higher_is_better": False,
    },
    "surface_dice_cc": {
        "builder": lambda cfg: CCSurfaceDiceMetric(
            cc_reduction=cfg["cc_reduction"],
            use_caching=False,
            class_thresholds=list(cfg["class_thresholds"]),
        ),
        "higher_is_better": True,
    },
}
 
class CCMetricsTracker(BaseMetricTracker):
    registry = CC_METRICS_REGISTRY
 
    def __init__(
        self,
        metric_names: Sequence[str],
        stages: Sequence[str] = ("valid", "test"),
        cc_reduction: str = "patient",
        aggregate_modes: Sequence[str] = ("patient",),
        worst_score: float = 30.0,
        class_thresholds: Sequence[float] = (1.0,),
        every_n_epochs: int = 1,
        as_volume: bool = True,
        skip_empty_gt: bool = True,
    ):
        super().__init__(
            metric_names,
            stages=stages,
            worst_score=worst_score,
            class_thresholds=class_thresholds,
            every_n_epochs=every_n_epochs,
            skip_empty_gt=skip_empty_gt,
        )
        self.aggregate_modes = aggregate_modes
        self.as_volume = as_volume
        self.cfg["cc_reduction"] = cc_reduction
 
    @staticmethod
    def _to_one_hot(mask_2d: torch.Tensor, as_volume: bool) -> torch.Tensor:
        fg = mask_2d.float().unsqueeze(0).unsqueeze(0)
        if as_volume:
            fg = fg.unsqueeze(2)
        return torch.cat([1.0 - fg, fg], dim=1)
 
    def _new_state(self) -> Dict[str, object]:
        return {name: self.registry[name]["builder"](self.cfg) for name in self.names}
 
    def _update_state(self, state, pred, gt) -> None:
        for b in range(pred.shape[0]):
            y = self._to_one_hot(gt[b, 0], self.as_volume)
            y_pred = self._to_one_hot(pred[b, 0], self.as_volume)
            for metric in state.values():
                metric(y_pred=y_pred, y=y)
 
    def _compute_state(self, state) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for name, metric in state.items():
            for mode in self.aggregate_modes:
                key = name if mode == "patient" else f"{name}_{mode}"
                try:
                    scores = metric.cc_aggregate(mode=mode)
                except Exception as exc:
                    warnings.warn(f"Failed aggregating {name} ({mode}): {exc}")
                    continue
                if scores is None or len(scores) == 0:
                    continue
                scores = torch.as_tensor(scores).float()
                scores = scores[~torch.isnan(scores)]
                if scores.numel() == 0:
                    continue
                results[key] = scores.mean().item()
        return results