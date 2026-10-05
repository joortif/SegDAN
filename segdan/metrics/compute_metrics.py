from typing import Sequence
import warnings
import torch

from segdan.metrics.basetracker import BaseMetricTracker
from segdan.metrics.monaimetrics import MONAI_METRICS_REGISTRY
from segdan.metrics.statmetrics import STAT_METRIC_FUNCTIONS, accuracy, iou_score, dice_score, precision, recall, f1_score
from segdan.metrics.custom_metric import custom_metric

metric_functions = {
    "accuracy": accuracy,
    "iou": iou_score,
    "dice": dice_score,
    "precision": precision,
    "recall": recall,
    "f1": f1_score,
}

def split_metric_names(metrics: Sequence[str]):
    stats, cc, unknown = [], [], []
    for name in metrics:
        name = str(name)
        
        if name in STAT_METRIC_FUNCTIONS:
            stats.append(name)
            
        elif name in CC_METRICS_REGISTRY:
            cc.append(name)

        elif name in MONAI_METRICS_REGISTRY:
            cc.append(name)
            
        else:
            unknown.append(name)
            
    if unknown:
        warnings.warn(f"Unknown metrics will be ignored: {unknown}")
        
    return stats, cc

def compute_metrics(results, metrics, classes, stage="train", trackers: Sequence[BaseMetricTracker] = ()):

    stat_names, _ = split_metric_names(metrics)

    tp = torch.cat([x["tp"] for x in results])
    fp = torch.cat([x["fp"] for x in results])
    fn = torch.cat([x["fn"] for x in results])
    tn = torch.cat([x["tn"] for x in results])

    results = {}
    out = {}

    for metric in stat_names:
        metric_fn = STAT_METRIC_FUNCTIONS[metric]

        score_global = custom_metric(tp, fp, fn, tn, metric_fn, reduction="micro")
        out[f"{metric}_{stage}"] = score_global.item() if torch.is_tensor(score_global) else score_global

        score_none = custom_metric(tp, fp, fn, tn, metric_fn, reduction="none")
        score_per_class = score_none.mean(dim=0)  
        
        if len(classes) == 2: 
            class_name = [c for c in classes if c != "background"][0]
            out[f"{metric}_{stage}_class_{class_name}"] = score_per_class.item()
        else:
            for class_idx, class_score in enumerate(score_per_class):
                class_name = classes[class_idx]
                out[f"{metric}_{stage}_class_{class_name}"] = class_score.item()
        
    for tracker in trackers:
        for name, value in tracker.compute(stage).items():
            out[f"{name}_{stage}"] = value
                
    return out
