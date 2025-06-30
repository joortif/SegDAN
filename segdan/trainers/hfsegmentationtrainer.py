from transformers import Trainer

import torch.nn.functional as F
import torch
import numpy as np

import segmentation_models_pytorch as smp

from segdan.metrics.custom_metric import custom_metric
from segdan.metrics.segmentationmetrics import dice_score

class HFSegmentationTrainer(Trainer):
    def __init__(self, *args, num_classes=None, ignore_index=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.binary = num_classes == 1

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)  

        loss = outputs.loss  
        
        return (loss, outputs) if return_outputs else loss
    
    def compute_metrics(eval_pred, num_classes, stage="train", ignore_index=255):
        predictions, label_ids = eval_pred
        
        class_logits = torch.from_numpy(predictions[0])      
        masks_q = torch.from_numpy(predictions[1])     
        
        probs = torch.softmax(class_logits, dim=-1)[..., 1]
        raw_mask = (probs[:, :, None, None] * masks_q).sum(dim=1)
        
        masks_tensor = torch.tensor(np.array(label_ids[0])).bool()  # (B, M, H, W)
        classes_tensor = torch.tensor(label_ids[1])

        gt_masks = []
        pred_masks = []

        for i in range(masks_tensor.shape[0]):
            gt_mask = torch.zeros_like(masks_tensor[i, 0])
            for m, c in zip(masks_tensor[i], classes_tensor[i]):
                if c.item() == 1:
                    gt_mask |= m
            gt_masks.append(gt_mask)

            pred = F.interpolate(
                raw_mask[i].unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
                size=gt_mask.shape,
                mode="bilinear",
                align_corners=False
            ).squeeze()
            pred_masks.append((pred.sigmoid() > 0.5).bool())
            
        pred_masks_array = torch.stack(pred_masks)
        gt_masks_array = torch.stack(gt_masks)

        tp, fp, fn, tn = smp.metrics.get_stats(pred_masks_array, gt_masks_array, mode="binary")

        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        iou_per_class = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
        dataset_accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        dataset_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        dataset_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
        dataset_f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        dataset_dice = custom_metric(tp, fp, fn, tn, dice_score, reduction="micro") 
        dice_per_class = custom_metric(tp, fp, fn, tn, dice_score, reduction="none") 

        metrics = {
            f"Accuracy": dataset_accuracy,
            f"Precision": dataset_precision,
            f"Recall": dataset_recall,
            f"F1_score": dataset_f1_score,
            f"Dice": dataset_dice,
            f"Iou": dataset_iou,
        }
        
        for class_idx in range(num_classes):
            iou_class = iou_per_class[class_idx]
            metrics[f"Class_{class_idx}_iou"] = iou_class.mean().item()  

            dice_class = dice_per_class[class_idx]
            metrics[f"Class_{class_idx}_dice"] = dice_class.mean().item()
        
        return metrics