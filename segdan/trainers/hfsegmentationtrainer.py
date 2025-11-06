import os

import logging
import torch
import numpy as np

import segmentation_models_pytorch as smp

import pandas as pd
from transformers import OneFormerForUniversalSegmentation, Trainer

import torch
import numpy as np
from types import SimpleNamespace
from typing import Optional, List, Tuple

from utils.utils import Utils
from segdan.metrics.compute_metrics import compute_metrics

logger = logging.getLogger(__name__)

class HFSegmentationTrainer(Trainer):
    def __init__(self, *args, num_classes=None, ignore_index=None,
                 test_dataset=None, processor=None, id2label=None, metrics_list=None, output_path=None, selection_metric=None, **kwargs):
        super().__init__(*args, **kwargs)
        if num_classes is None:
            raise ValueError("You must pass num_classes to HFSegmentationTrainer.")
        self.ignore_index = ignore_index if ignore_index is not None else 255
        self.num_classes = num_classes
        self.test_dataset = test_dataset
        self.processor = processor
        self.id2label = id2label or {i: str(i) for i in range(num_classes)}
        self.classes = [self.id2label[i] for i in range(num_classes)]
        self.metrics_list = metrics_list or ["iou", "dice", "precision", "recall"]
        self.selection_metric = selection_metric
        self.output_path = output_path

    def _labels_to_numpy_list_and_sizes(self, labels_in, class_labels_in=None) -> Tuple[List[np.ndarray], Optional[List[Tuple[int,int]]]]:
        if labels_in is None:
            return [], None

        labels_np = []

        if torch.is_tensor(labels_in):
            arr = labels_in.detach().cpu().numpy()
            labels_in = arr

        if isinstance(labels_in, np.ndarray):
            if labels_in.ndim == 3:
                for i in range(labels_in.shape[0]):
                    labels_np.append(labels_in[i].astype(np.int64))
            elif labels_in.ndim == 2:
                labels_np.append(labels_in.astype(np.int64))
            elif labels_in.ndim == 4:
                B, M, H, W = labels_in.shape
                if class_labels_in is not None:
                    for i in range(B):
                        map_i = np.full((H, W), fill_value=self.ignore_index, dtype=np.int64)
                        cls_list = None
                        if torch.is_tensor(class_labels_in):
                            cls_list = class_labels_in.detach().cpu().numpy()
                        else:
                            cls_list = np.asarray(class_labels_in)
                        if cls_list.shape[0] == B:
                            cls_ids = cls_list[i]
                        else:
                            cls_ids = class_labels_in[i]
                        for j in range(M):
                            cls_id = int(cls_ids[j])
                            mask = labels_in[i, j]
                            mask_bool = mask > 0.5
                            map_i[mask_bool] = cls_id
                        labels_np.append(map_i)
                else:
                    for i in range(B):
                        map_i = labels_in[i].argmax(axis=0).astype(np.int64)
                        labels_np.append(map_i)
            else:
                try:
                    for i in range(labels_in.shape[0]):
                        labels_np.append(labels_in[i].astype(np.int64))
                except Exception:
                    pass
            sizes = [tuple(x.shape[-2:]) for x in labels_np] if labels_np else None
            return labels_np, sizes

        if isinstance(labels_in, (list, tuple)):
            for idx, item in enumerate(labels_in):
                if torch.is_tensor(item):
                    arr = item.detach().cpu().numpy()
                    if arr.ndim == 2:
                        labels_np.append(arr.astype(np.int64))
                    elif arr.ndim == 3:
                        if arr.shape[0] == 1:
                            labels_np.append(arr[0].astype(np.int64))
                        else:
                            labels_np.append(arr.argmax(axis=0).astype(np.int64))
                elif isinstance(item, np.ndarray):
                    arr = item
                    if arr.ndim == 2:
                        labels_np.append(arr.astype(np.int64))
                    elif arr.ndim == 3:
                        if arr.shape[0] > 1:
                            labels_np.append(arr.argmax(axis=0).astype(np.int64))
                        else:
                            labels_np.append(arr[0].astype(np.int64))
                else:
                    try:
                        arr = np.asarray(item)
                        if arr.ndim == 2:
                            labels_np.append(arr.astype(np.int64))
                        elif arr.ndim == 3:
                            labels_np.append(arr.argmax(axis=0).astype(np.int64))
                    except Exception:
                        continue
            sizes = [tuple(x.shape[-2:]) for x in labels_np] if labels_np else None
            return labels_np, sizes

        return [], None

    def _outputs_to_postprocess_input(self, outputs, device):
        cls = getattr(outputs, "class_queries_logits", None)
        masks = getattr(outputs, "masks_queries_logits", None)

        if cls is None or masks is None:
            if isinstance(outputs, dict):
                cls = outputs.get("class_queries_logits")
                masks = outputs.get("masks_queries_logits")

        if isinstance(cls, np.ndarray):
            cls = torch.from_numpy(cls)
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)

        if torch.is_tensor(cls):
            cls = cls.to(device=device)
        if torch.is_tensor(masks):
            masks = masks.to(device=device)

        return SimpleNamespace(class_queries_logits=cls, masks_queries_logits=masks)

    def _preds_list_to_cpu_tensor(self, preds_list) -> Optional[torch.Tensor]:
        if preds_list is None or len(preds_list) == 0:
            return None

        normalized = []
        for p in preds_list:
            if torch.is_tensor(p):
                pt = p.detach().cpu().long()
            else:
                arr = np.asarray(p)
                pt = torch.from_numpy(arr).long()
            if pt.ndim == 3 and pt.shape[0] == 1:
                pt = pt[0]
            normalized.append(pt)
        try:
            stacked = torch.stack(normalized, dim=0)
        except Exception:
            stacked = None
        return stacked

    def _flatten_preds_labels_from_trainer_output(self, predictions, label_ids):
        labels_list = []
        if isinstance(label_ids, np.ndarray):
            if label_ids.ndim == 3:
                for i in range(label_ids.shape[0]):
                    labels_list.append(label_ids[i])
            else:
                labels_list.append(label_ids)
        elif isinstance(label_ids, (list, tuple)):
            for part in label_ids:
                if isinstance(part, np.ndarray):
                    if part.ndim == 3:
                        for i in range(part.shape[0]):
                            labels_list.append(part[i])
                    elif part.ndim == 2:
                        labels_list.append(part)
                elif torch.is_tensor(part):
                    arr = part.detach().cpu().numpy()
                    if arr.ndim == 3:
                        for i in range(arr.shape[0]):
                            labels_list.append(arr[i])
                    else:
                        labels_list.append(arr)
                else:
                    try:
                        arr = np.asarray(part)
                        if arr.ndim == 3:
                            for i in range(arr.shape[0]):
                                labels_list.append(arr[i])
                        else:
                            labels_list.append(arr)
                    except Exception:
                        continue
        else:
            raise RuntimeError("Unknown label_ids type: %s" % type(label_ids))

        preds_list = []
        if isinstance(predictions, dict) and "preds" in predictions:
            p = predictions["preds"]
            if isinstance(p, list):
                for batch_part in p:
                    if isinstance(batch_part, list):
                        preds_list.extend(batch_part)
                    elif isinstance(batch_part, np.ndarray):
                        if batch_part.ndim == 3:
                            for i in range(batch_part.shape[0]):
                                preds_list.append(batch_part[i])
                        elif batch_part.ndim == 2:
                            preds_list.append(batch_part)
                        else:
                            preds_list.append(batch_part)
                    else:
                        try:
                            arr = np.asarray(batch_part)
                            if arr.ndim == 3:
                                for i in range(arr.shape[0]):
                                    preds_list.append(arr[i])
                            else:
                                preds_list.append(arr)
                        except Exception:
                            continue
            elif isinstance(p, np.ndarray):
                if p.ndim == 3:
                    for i in range(p.shape[0]):
                        preds_list.append(p[i])
                elif p.ndim == 2:
                    preds_list.append(p)
            else:
                try:
                    arr = np.asarray(p)
                    if arr.ndim == 3:
                        for i in range(arr.shape[0]):
                            preds_list.append(arr[i])
                    else:
                        preds_list.append(arr)
                except Exception:
                    pass
        elif isinstance(predictions, np.ndarray):
            if predictions.ndim == 3:
                for i in range(predictions.shape[0]):
                    preds_list.append(predictions[i])
            else:
                preds_list.append(predictions)
        elif isinstance(predictions, list):
            for item in predictions:
                if isinstance(item, np.ndarray) and item.ndim == 3:
                    for i in range(item.shape[0]):
                        preds_list.append(item[i])
                elif isinstance(item, np.ndarray) and item.ndim == 2:
                    preds_list.append(item)
                elif isinstance(item, list):
                    for p in item:
                        preds_list.append(np.asarray(p))
                else:
                    preds_list.append(np.asarray(item))
        else:
            try:
                arr = np.asarray(predictions)
                if arr.ndim == 3:
                    for i in range(arr.shape[0]):
                        preds_list.append(arr[i])
                else:
                    preds_list.append(arr)
            except Exception:
                raise RuntimeError("Unsupported predictions type: %s" % type(predictions))

        min_len = min(len(preds_list), len(labels_list))
        if len(preds_list) != len(labels_list):
            logger.warning(f"WARNING: predictions ({len(preds_list)}) != labels ({len(labels_list)}), truncating to {min_len}")
            preds_list = preds_list[:min_len]
            labels_list = labels_list[:min_len]

        return preds_list, labels_list

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        device = next(model.parameters()).device
        
        inputs_model = {
            "pixel_values":inputs.get("pixel_values"),
            "pixel_mask":inputs.get("pixel_mask"),
            "mask_labels":inputs.get("mask_labels"),
            "class_labels":inputs.get("class_labels"),
            "return_dict":True,
        }

        if isinstance(model, OneFormerForUniversalSegmentation):
            inputs_model["task_inputs"]=inputs.get("task_inputs")
        
        with torch.no_grad():
            outputs = model(
                **inputs_model
            )

        loss = getattr(outputs, "loss", None)
        
        if loss.dim() > 0:
            loss = loss.mean()
        
        labels_np_list, target_sizes_from_labels = self._labels_to_numpy_list_and_sizes(
            inputs.get("mask_labels"), class_labels_in=inputs.get("class_labels")
        )

        model_out_for_post = self._outputs_to_postprocess_input(outputs, device)

        try:
            logits_batch = model_out_for_post.class_queries_logits.shape[0]
        except Exception:
            logits_batch = None

        target_sizes = target_sizes_from_labels
        if target_sizes is None:
            pv = inputs.get("pixel_values")
            if torch.is_tensor(pv):
                target_sizes = [(pv.shape[-2], pv.shape[-1]) for _ in range(pv.shape[0])]
            else:
                target_sizes = None

        if logits_batch is not None and target_sizes is not None:
            if len(target_sizes) != logits_batch:
                if len(target_sizes) > logits_batch:
                    target_sizes = target_sizes[:logits_batch]
                else:
                    last = target_sizes[-1]
                    while len(target_sizes) < logits_batch:
                        target_sizes.append(last)        

        try:
            preds_list = self.processor.post_process_semantic_segmentation(model_out_for_post, target_sizes=target_sizes)
        except Exception as e:
            logger.error("post_process_semantic_segmentation failed in prediction_step:", e)
            return (loss, None, None)
        
        preds_batch_tensor = self._preds_list_to_cpu_tensor(preds_list)

        labels_batch_tensor = None
        if labels_np_list:
            try:
                labels_batch_tensor = torch.stack([torch.from_numpy(x).long() for x in labels_np_list], dim=0)
            except Exception:
                tmp = []
                for x in labels_np_list:
                    tmp.append(torch.from_numpy(x).long())
                labels_batch_tensor = torch.stack(tmp, dim=0) if tmp else None

        if preds_batch_tensor is not None and labels_batch_tensor is not None:
            if preds_batch_tensor.shape[0] != labels_batch_tensor.shape[0]:
                min_b = min(preds_batch_tensor.shape[0], labels_batch_tensor.shape[0])
                preds_batch_tensor = preds_batch_tensor[:min_b]
                labels_batch_tensor = labels_batch_tensor[:min_b]

        return (loss, {"preds": preds_batch_tensor}, labels_batch_tensor)

    def compute_metrics_huggingface_from_pred(self, predictions, label_ids):
        
        preds_list, labels_list = self._flatten_preds_labels_from_trainer_output(predictions, label_ids)

        if len(preds_list) == 0 or len(labels_list) == 0:
            return {}

        device = next(self.model.parameters()).device
        preds_tensor = torch.stack([torch.from_numpy(p).long() for p in preds_list], dim=0).to(device)
        labels_tensor = torch.stack([torch.from_numpy(l).long() for l in labels_list], dim=0).to(device)
        
        if self.num_classes == 2:
            mode = "binary"
            out = preds_tensor.unsqueeze(1).long()  # (N,1,H,W)
            tgt = labels_tensor.unsqueeze(1).long()  # (N,1,H,W)

            num_classes=None
            ignore_idx = self.ignore_index if getattr(self, "ignore_index", None) is not None else None
            if ignore_idx is not None:
                ignore_idx = None
        else:
            mode = "multiclass"
            out = preds_tensor.long()
            tgt = labels_tensor.long()
            ignore_idx = self.ignore_index
            num_classes = self.num_classes
                   
        tp, fp, fn, tn = smp.metrics.get_stats(out, tgt, mode=mode, ignore_index=ignore_idx, num_classes=num_classes)
        
        results = []
        N = tp.shape[0]
        C = tp.shape[1]
        for i in range(N):
            results.append({
                "tp": tp[i].unsqueeze(0),  
                "fp": fp[i].unsqueeze(0),
                "fn": fn[i].unsqueeze(0),
                "tn": tn[i].unsqueeze(0),
            })
        
        metrics_result = compute_metrics(results, self.metrics_list, self.classes, stage="eval")
        return metrics_result

    def evaluate(self, eval_dataset=None, **kwargs):
        return super().evaluate(eval_dataset=eval_dataset, **kwargs)

    def test(self):
        test_results = self.evaluate(eval_dataset=self.test_dataset)
        return test_results
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)

        loss = getattr(outputs, "loss", None)
        if loss is None and isinstance(outputs, dict):
            loss = outputs.get("loss")

        if not torch.is_tensor(loss):
            loss = torch.tensor(loss, device=next(model.parameters()).device, dtype=torch.float)

        device = next(model.parameters()).device
        loss = loss.to(device)

        if loss.dim() > 0:
            loss = loss.mean()

        if return_outputs:
            return loss, outputs
        return loss
    
    def save_metrics(self, metrics, experiment_name, filename, hf=False, training_time=None):
        if not metrics:
            logger.info("No metrics to save.")
            return metrics

        metrics_dict = metrics  
        if hf:
            metrics_dict = Utils.adapt_hf_metrics(metrics_dict)
        
        df = pd.DataFrame([metrics_dict])  
        df.insert(0, "Experiment", experiment_name)  
        
        if training_time is not None:
            df["Training Time (min)"] = round(training_time / 60.0, 2)

        if os.path.exists(filename):
            df_existing = pd.read_csv(filename, sep=';')
            df_combined = pd.concat([df_existing, df], ignore_index=True)
        else:
            df_combined = df

        file_output_path = os.path.join(os.path.dirname(self.output_path), filename)
        if os.path.exists(file_output_path):
            df.to_csv(file_output_path, sep=';', mode='a', header=False, index=False)
        else:
            df.to_csv(file_output_path, sep=';', index=False)  

        logger.info(f"Metrics saved in file {file_output_path}")

        evaluation_metric = metrics_dict.get(f"{self.selection_metric}_test")
        return evaluation_metric