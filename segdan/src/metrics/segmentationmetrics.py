import numpy as np
from sklearn.metrics import precision_score, jaccard_score, f1_score
import torch
import warnings

import os
import pandas as pd

def _handle_zero_division(x, zero_division):
    nans = torch.isnan(x)
    if torch.any(nans) and zero_division == "warn":
        warnings.warn("Zero division in metric calculation!")
    elif zero_division == "ignore":
        return x[~nans]
    
    value = zero_division if zero_division != "warn" else 0
    value = torch.tensor(value, dtype=x.dtype).to(x.device)
    x = torch.where(nans, value, x)
    return x

def accuracy(tp, fp, fn, tn):
    return (tp + tn) / (tp + fp + fn + tn)

def iou_score(tp, fp, fn, tn):
    return tp / (tp + fp + fn)

def dice_score(tp, fp, fn, tn):
    return (2 * tp) / (2 * tp + fp + fn)

def precision(tp, fp, fn, tn):
    return tp / (tp + fp)

def recall(tp, fp, fn, tn):
    return tp / (tp + fn)

def f1_score(tp, fp, fn, tn, beta=1):
    beta_tp = (1 + beta**2) * tp
    beta_fn = (beta**2) * fn
    score = beta_tp / (beta_tp + beta_fn + fp)
    return score

def custom_metric(tp, fp, fn, tn, metric_fn, reduction="micro", zero_division="warn"):
    if reduction == "micro":
        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()
        tn = tn.sum()
        score = metric_fn(tp, fp, fn, tn)

    elif reduction == "micro-imagewise":
        tp = tp.sum(1)
        fp = fp.sum(1)
        fn = fn.sum(1)
        tn = tn.sum(1)
        score = metric_fn(tp, fp, fn, tn)
        score = _handle_zero_division(score, zero_division)
        score = score.mean()
        
    elif reduction == "none" or reduction is None:
        score = metric_fn(tp, fp, fn, tn)
        score = _handle_zero_division(score, zero_division)
        
    return score

def save_metrics(trainer,dataloader,experiment_name,filename="metrics.csv",training_time=None):
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