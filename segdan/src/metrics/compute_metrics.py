import torch
import pandas as pd
import os

from src.metrics.segmentationmetrics import accuracy, iou_score, dice_score, precision, recall, f1_score
from src.metrics import custom_metric

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

def save_metrics_smp(trainer,dataloader,experiment_name,filename="metrics.csv",training_time=None):
    metrics = trainer.evaluate(eval_dataset=dataloader)
    
    if not metrics:
        print("No metrics to save.")
        return metrics

    df = pd.DataFrame(metrics)
    df.insert(0, experiment_name)

    if training_time is not None:
        df["Training Time (min)"] = round(training_time / 60.0, 2)

    if os.path.exists(filename):
        df_existing = pd.read_csv(filename, sep=';')
        df_combined = pd.concat([df_existing, df], ignore_index=True)
    else:
        df_combined = df

    df_combined.to_csv(filename, sep=';', index=False)
    print(f"Métricas guardadas en {filename}")
    return metrics