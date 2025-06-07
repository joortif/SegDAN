from functools import partial
import time

from transformers import Mask2FormerForUniversalSegmentation, MaskFormerForInstanceSegmentation, OneFormerForUniversalSegmentation, MaskFormerImageProcessor, OneFormerImageProcessor, OneFormerProcessor
from transformers import TrainingArguments

import torch
import torch.nn.functional as F

import segmentation_models_pytorch as smp
import numpy as np

import pandas as pd

from src.metrics.compute_metrics import compute_metrics
from src.models.semanticsegmentationmodel import SemanticSegmentationModel
from src.trainers.hfsegmentationtrainer import HFSegmentationTrainer

import os

class HFTransformerModel(SemanticSegmentationModel):

    MODEL_CONFIGS = {
        "maskformer": {
            "model_class": MaskFormerForInstanceSegmentation,
            "base_name": "facebook/maskformer-swin-{size}-ade",
            "processor_class": MaskFormerImageProcessor,
        },
        "mask2former": {
            "model_class": Mask2FormerForUniversalSegmentation,
            "base_name": "facebook/mask2former-swin-{size}-ade-semantic",
            "processor_class": MaskFormerImageProcessor,
        },
        "oneformer": {
            "model_class": OneFormerForUniversalSegmentation,
            "base_name": "shi-labs/oneformer_ade20k_swin_{size}",
            "processor_class": OneFormerImageProcessor,
        },
    }

    def __init__(self, model_name, model_size, out_classes, metrics, epochs, batch_size, ignore_index=255):
        super.__init__(out_classes, epochs, metrics, ignore_index, model_name, model_size)
        if self.model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported HuggingFace semantic segmentation model {self.model_name}. Supported models are: {list(self.MODEL_CONFIGS.keys())}")
        
        config = self.MODEL_CONFIGS[self.model_name]
        pretrained_name = config["base_name"].format(size=self.model_size)
        
        self.model = config["model_class"].from_pretrained(pretrained_name, num_labels=self.out_classes, ignore_mismatched_sizes=True)
        self.feature_extractor = config["processor_class"].from_pretrained(pretrained_name, do_resize=False, use_fast=True)

        self.batch_size = batch_size

    def huggingface_collate_fn(self, batch):
        images, masks = zip(*batch)
        
        images = [img.transpose(1, 2, 0) for img in images]

        kwargs = {
            "images": images,
            "segmentation_maps": masks,
            "ignore_index": 255,
            "return_tensors": "pt",
            "do_resize": False,
        }

        if isinstance(self.feature_extractor, OneFormerProcessor):
            kwargs["task_inputs"] = ["semantic"] * len(images)
            self.feature_extractor.image_processor.num_text = 65

        encoded_inputs = self.feature_extractor(**kwargs)
        
        result =  {
            "pixel_values": encoded_inputs["pixel_values"],
            "pixel_mask": encoded_inputs.get("pixel_mask"),
            "mask_labels": encoded_inputs.get("mask_labels"),
            "class_labels": encoded_inputs.get("class_labels"),
        }
        
        if "task_inputs" in encoded_inputs:
            result["task_inputs"] = encoded_inputs["task_inputs"]
            
        return result

    def compute_semantic_metrics(self, eval_pred, stage):
        predictions, label_ids = eval_pred
        
        class_logits = torch.from_numpy(predictions[0])      
        masks_q      = torch.from_numpy(predictions[1])    
        
        masks_tensor = torch.tensor(np.array(label_ids[0][0])).bool() 
        classes_tensor = torch.tensor(np.array(label_ids[1][0]))
        
        batch_size = class_logits.shape[0]
        _, H_gt, W_gt = masks_tensor.shape
        
        if self.out_classes == 1:
            probs = torch.softmax(class_logits, dim=-1)[..., 1]
            raw_mask = (probs[:, :, None, None] * masks_q).sum(dim=1)

            gt_masks = []
            pred_masks = []

            for i in range(batch_size):
                gt_mask = torch.zeros_like(masks_tensor[i, 0])
                for m, c in zip(masks_tensor, classes_tensor):
                    if c.item() == 1:
                        gt_mask |= m
                gt_masks.append(gt_mask)

                pred = F.interpolate(
                    raw_mask[i].unsqueeze(0).unsqueeze(0),  
                    size=gt_mask.shape,
                    mode="bilinear",
                    align_corners=False
                ).squeeze()
                pred_masks.append((pred.sigmoid() > 0.5).bool())
            
            pred_masks_array = torch.stack(pred_masks)
            gt_masks_array = torch.stack(gt_masks)
                
            tp, fp, fn, tn = smp.metrics.get_stats(pred_masks_array, gt_masks_array, mode="binary")
        
        else:     
            probs = torch.softmax(class_logits, dim=-1)
            raw_mask = torch.einsum('bmc,bmhw->bchw', probs, masks_q)
            
            gt_masks_array = torch.zeros((batch_size, self.out_classes, H_gt, W_gt), dtype=torch.bool)

            for i in range(batch_size):
                for m, c in zip(masks_tensor, classes_tensor):
                    m = m.float().unsqueeze(0).unsqueeze(0)
                    m = F.interpolate(m, size=(H_gt, W_gt), mode="nearest").squeeze().bool()
                    gt_masks_array[i, c] |= m

            pred_classes_mask = raw_mask.argmax(dim=1)  
            gt_classes_mask = gt_masks_array.float().argmax(dim=1)  
            
            pred_classes_mask = F.interpolate(
                pred_classes_mask.unsqueeze(1).float(),  
                size=(gt_classes_mask.shape[-2], gt_classes_mask.shape[-1]),
                mode="nearest"
            ).squeeze(1).long()

            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_classes_mask, gt_classes_mask,
                mode="multiclass",
                num_classes=self.out_classes,
                ignore_index=self.ignore_index
            )
        
        statistics = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }

        metric_results = compute_metrics(statistics, self.metrics, stage)
        
        return metric_results
    
    def save_metrics_hf(trainer,dataloader,experiment_name,filename="metrics.csv",training_time=None):
        metrics = trainer.predict(test_dataset=dataloader)
        
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
        print(f"Metrics saved in {filename}")
        return metrics

    def train(self, train_dataset, valid_dataset, test_dataset, output_path):

        training_args = TrainingArguments(
        output_dir=output_path,          
        eval_strategy="epoch",
        learning_rate=5e-5,             
        per_device_train_batch_size=self.batch_size,   
        per_device_eval_batch_size=self.batch_size,    
        num_train_epochs=self.epochs,              
        weight_decay=0.01,               
        logging_dir=None,            
        logging_strategy="no",                
        save_strategy="epoch",
        save_total_limit=3,
        eval_steps=50,
        fp16=True,
        dataloader_num_workers=4
        )
        
        trainer = HFSegmentationTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            num_classes=len(train_dataset.get_class_map())-1, 
            ignore_index=255,
            dice_loss_kwargs={"from_logits":True},
            data_collator = self.huggingface_collate_fn,
            compute_metrics=partial(self.compute_semantic_metrics, stage="val")

        )

        start_time = time.time()
        trainer.train()
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total training time: {total_time / 60:.2f} minutes")
        
        valid_metrics = trainer.evaluate(eval_dataset=valid_dataset)
        print(valid_metrics)
        
        if test_dataset:
            trainer.compute_metrics = partial(self.compute_semantic_metrics, stage="test")
            test_metrics = self.save_metrics_hf(trainer, test_dataset, f"{self.model_name.capitalize()} - {self.model_size.capitalize()}", os.path.join(output_path, "metrics.csv"), mode="test", training_time=total_time)

        return {
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "training_time_min": total_time,
    }
        
    



