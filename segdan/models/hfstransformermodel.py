import os
import logging

from functools import partial
import time
import transformers
from transformers import Mask2FormerForUniversalSegmentation, MaskFormerForInstanceSegmentation, OneFormerForUniversalSegmentation, MaskFormerImageProcessor, OneFormerImageProcessor, OneFormerProcessor
from transformers import TrainingArguments

from packaging import version
import torch

from models.callbacks import SaveWeightsCallbackHF
from segdan.models.semanticsegmentationmodel import SemanticSegmentationModel
from segdan.trainers.hfsegmentationtrainer import HFSegmentationTrainer

logger = logging.getLogger(__name__)

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

    def __init__(self, model_name, model_size, classes, metrics, selection_metric, epochs, imgsz, output_path, fraction):
        super.__init__(self, classes=classes, epochs=epochs, imgsz=imgsz, metrics=metrics, selection_metric=selection_metric, 
                       model_name=model_name, model_size=model_size, output_path=output_path, fraction=fraction)
        
        if self.model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported HuggingFace semantic segmentation model {self.model_name}. Supported models are: {list(self.MODEL_CONFIGS.keys())}")
        
        config = self.MODEL_CONFIGS[self.model_name]
        pretrained_name = config["base_name"].format(size=self.model_size)
        
        self.model = config["model_class"].from_pretrained(pretrained_name, num_labels=self.out_classes, ignore_mismatched_sizes=True)
        self.feature_extractor = config["processor_class"].from_pretrained(pretrained_name, do_resize=False, use_fast=True)
        self.id2label = {i:classes[i] for i in range(len(classes))}

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
            self.feature_extractor.image_processor.num_text = 1

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

    def init_trainer(self, train_dataset, valid_dataset, test_dataset):

        ver = transformers.__version__
        strategy_key = (
            "evaluation_strategy" if version.parse(ver) >= version.parse("4.29.0") else "eval_strategy"
        )

        training_args = TrainingArguments(
            output_dir=self.output_path,          
            **{strategy_key: "epoch"},
            learning_rate=5e-5, #self.lr ??            
            per_device_train_batch_size=self.batch,   
            per_device_eval_batch_size=self.batch,    
            num_train_epochs=self.epochs,              
            weight_decay=0.01,               
            logging_dir=None,            
            logging_strategy="no",                
            save_strategy="no",
            save_total_limit=3,
            label_names=["mask_labels"],
            fp16=True,
            remove_unused_columns=False,
            report_to="none"
        )

        data_collator = partial(self.huggingface_collate_fn, processor=self.feature_extractor)
        
        self.trainer = HFSegmentationTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            test_dataset=test_dataset,
            num_classes = len(self.id2label),
            processor=self.feature_extractor,
            id2label = self.id2label,
            metric = self.metrics,
            selection_metric = self.selection_metric,
            data_collator = data_collator,
            output_path = self.output_path
        )

        def compute_metrics_wrapper(eval_pred):
            preds = getattr(eval_pred, "predictions", None)
            labels = getattr(eval_pred, "label_ids", None)

            return self.trainer.compute_metrics_huggingface_from_pred(preds, labels)
        
        self.trainer.compute_metrics = compute_metrics_wrapper


    def run_training(self, save_model=False, save_n_ckpts=10, save_last_epochs=False):

        self.trainer.add_callback(SaveWeightsCallbackHF(save_n_ckpts=save_n_ckpts, save_last_epochs=save_last_epochs, output_path=self.output_path))

        start_time = time.time()
        self.trainer.train()
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total training time: {total_time / 60:.2f} minutes")
        

        if self.trainer.eval_dataset is not None:
            self.trainer.evaluate()

        os.makedirs(self.output_path, exist_ok=True)

        model_name = self.trainer.model.__class__.__name__
        
        
        test_metrics = self.trainer.test()
        results_csv_path = f"{model_name}.csv"
        row_name = f"{model_name}_{self.trainer.self.imgsz}x{self.trainer.self.imgsz}_b{self.trainer.batch_size}"
        evaluation_metric = self.trainer.save_metrics(metrics=test_metrics, experiment_name=row_name, hf=True, filename=results_csv_path, training_time=total_time)
        
        weights_path = os.path.join(self.output_path, f"{self.trainer.model.__class__.__name__}_weights.pt")
        model_output_path = torch.save(self.trainer.model.state_dict(), weights_path)

        return evaluation_metric, model_output_path
        
    



