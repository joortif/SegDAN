from transformers import Trainer

import torch.nn.functional as F
import torch

import segmentation_models_pytorch as smp

from src.metrics.segmentationmetrics import custom_metric, dice_score

class HFSegmentationTrainer(Trainer):
    def __init__(self, *args, num_classes=None, ignore_index=None, dice_loss_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.criterion = self.get_dice_loss(num_classes, **(dice_loss_kwargs or {}))
        self.binary = num_classes == 1
        
    def get_dice_loss(self, num_classes, from_logits=True, **kwargs):
        if num_classes == 1 or num_classes == 2:
            return smp.losses.DiceLoss(mode="binary", from_logits=from_logits, **kwargs)
        else:
            return smp.losses.DiceLoss(mode="multiclass", from_logits=from_logits, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("mask_labels")
        
        outputs = model(**inputs)

        logits = outputs.class_queries_logits         # [B, Q, C]
        pred_masks = outputs.masks_queries_logits     # [B, Q, H/4, W/4]

        class_probs = logits.softmax(dim=-1)
        weighted_masks = torch.einsum('bqc,bqhw->bchw', class_probs, pred_masks)

        if self.binary:
            weighted_masks = weighted_masks[:, 1:2, :, :]  
        
        labels = F.interpolate(labels.float(), size=weighted_masks.shape[-2:], mode='bilinear', align_corners=False)

        loss = self.criterion(weighted_masks, labels)

        return (loss, outputs) if return_outputs else loss

    def compute_metrics(self, eval_pred):
        logits = torch.tensor(eval_pred.predictions[0])          # [B, Q, C]
        pred_masks = torch.tensor(eval_pred.predictions[1])      # [B, Q, H/4, W/4]
        labels = torch.tensor(eval_pred.label_ids[0])            # [B, C - 1, H, W]
                    
        class_probs = logits.softmax(dim=-1)
        weighted_masks = torch.einsum('bqc,bqhw->bchw', class_probs, pred_masks)  # [B, C, H, W]
        pred_mask = weighted_masks.argmax(dim=1)
        
        labels = labels.squeeze(1) if labels.shape[1] == 1 else labels.argmax(dim=1)
                    
        if pred_mask.shape[-2:] != labels.shape[-2:]:
            pred_mask = F.interpolate(pred_mask.unsqueeze(1).float(), size=labels.shape[-2:], mode="nearest").squeeze(1).long()
            
        if self.binary:
            metric_args = {"mode": "binary"}
        else:
            metric_args = {"mode": "multiclass", "num_classes": self.number_of_classes}

        if self.ignore_index is not None and not self.binary:
            metric_args["ignore_index"] = self.ignore_index

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), labels.long(), **metric_args)

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
        
        for class_idx in range(self.num_classes):
            iou_class = iou_per_class[class_idx]
            metrics[f"Class_{class_idx}_iou"] = iou_class.mean().item()  

            dice_class = dice_per_class[class_idx]
            metrics[f"Class_{class_idx}_dice"] = dice_class.mean().item()
        
        return metrics