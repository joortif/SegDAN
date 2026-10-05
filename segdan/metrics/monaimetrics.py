from typing import Dict, List
 
import torch
from monai.metrics import (
    compute_average_surface_distance,
    compute_hausdorff_distance,
    compute_surface_dice,
)

from segdan.metrics.basetracker import BaseMetricTracker

MONAI_METRICS_REGISTRY = {
    "hd": {
        "fn": lambda p, g, cfg: compute_hausdorff_distance(p, g, include_background=True),
        "higher_is_better": False,
    },
    "hd95": {
        "fn": lambda p, g, cfg: compute_hausdorff_distance(
            p, g, include_background=True, percentile=95
        ),
        "higher_is_better": False,
    },
    "assd": {
        "fn": lambda p, g, cfg: compute_average_surface_distance(
            p, g, include_background=True, symmetric=True
        ),
        "higher_is_better": False,
    },
    "surface_dice": {
        "fn": lambda p, g, cfg: compute_surface_dice(
            p, g, class_thresholds=list(cfg["class_thresholds"]), include_background=True
        ),
        "higher_is_better": True,
    },
}

class MonaiMetricsTracker(BaseMetricTracker):
    registry = MONAI_METRICS_REGISTRY
 
    def _new_state(self) -> Dict[str, List[torch.Tensor]]:
        return {name: [] for name in self.names}
 
    def _update_state(self, state, pred, gt) -> None:
        for name in self.names:
            scores = self.registry[name]["fn"](pred, gt, self.cfg)  
            state[name].append(scores[:, 0].float())
 
    def _compute_state(self, state) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for name, chunks in state.items():
            scores = torch.cat(chunks)

            scores[torch.isinf(scores)] = self.cfg["worst_score"]
            scores = scores[~torch.isnan(scores)]
            if scores.numel() > 0:
                results[name] = scores.mean().item()
        return results