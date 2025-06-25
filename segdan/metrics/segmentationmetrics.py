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

