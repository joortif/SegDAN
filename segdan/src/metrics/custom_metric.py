import torch
import warnings

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