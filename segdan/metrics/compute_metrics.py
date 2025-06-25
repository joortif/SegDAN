import torch

from metrics.segmentationmetrics import accuracy, iou_score, dice_score, precision, recall, f1_score
from metrics import custom_metric

metric_functions = {
    "accuracy": accuracy,
    "iou": iou_score,
    "dice": dice_score,
    "precision": precision,
    "recall": recall,
    "f1": f1_score,
}

def compute_metrics(results, metrics, stage="train"):
    tp = torch.cat([x["tp"] for x in results])
    fp = torch.cat([x["fp"] for x in results])
    fn = torch.cat([x["fn"] for x in results])
    tn = torch.cat([x["tn"] for x in results])

    results = {}

    for metric in metrics:
        metric_fn = metric_functions[metric]

        score_global = custom_metric(tp, fp, fn, tn, metric_fn, reduction="micro")
        results[f"{metric}_{stage}"] = score_global.item() if torch.is_tensor(score_global) else score_global

        score_per_class = custom_metric(tp, fp, fn, tn, metric_fn, reduction="none")
        if torch.is_tensor(score_per_class):
            for class_idx, class_score in enumerate(score_per_class):
                results[f"{metric}_{stage}_class_{class_idx}"] = class_score.item()
        else:
            for class_idx, class_score in enumerate(score_per_class):
                results[f"{metric}_{stage}_class_{class_idx}"] = class_score

    return results