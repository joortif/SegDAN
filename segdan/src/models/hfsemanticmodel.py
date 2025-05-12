import time
from transformers import Mask2FormerForUniversalSegmentation, MaskFormerForInstanceSegmentation, OneFormerForUniversalSegmentation, MaskFormerImageProcessor, OneFormerImageProcessor
from transformers import TrainingArguments

from segdan.src.metrics.segmentationmetrics import save_metrics
from src.trainers.hfsegmentationtrainer import HFSegmentationTrainer

import os

class HFFormerModel():

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

    def __init__(self, model_name, model_size, out_classes, metrics, selection_metric, epochs, batch_size):
        self.model_name = model_name.lower()
        self.model_size = model_size.lower()

        if self.model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported HuggingFace semantic segmentation model {self.model_name}. Supported models are: {list(self.MODEL_CONFIGS.keys())}")
        
        config = self.MODEL_CONFIGS[self.model_name]
        pretrained_name = config["base_name"].format(size=self.model_size)
        
        self.model = config["model_class"].from_pretrained(pretrained_name, num_labels=out_classes, ignore_mismatched_sizes=True)
        self.feature_extractor = config["processor_class"].from_pretrained(pretrained_name, do_resize=False)

        self.metrics = metrics
        self.selection_metric = selection_metric
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, train_dataset, valid_dataset, test_dataset, output_path):

        training_args = TrainingArguments(
        output_dir="./results_hf",          
        eval_strategy="steps",
        eval_steps=100, 
        learning_rate=5e-5,  #Probar otro lr          
        per_device_train_batch_size=self.batch_size,   
        per_device_eval_batch_size=self.batch_size,    
        num_train_epochs=self.epochs,              
        weight_decay=0.01,               
        logging_dir=None,            
        logging_strategy="no",                
        save_strategy="epoch",
        save_total_limit=3,
        fp16=True,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",     # Cargar el mejor modelo al final del entrenamiento
        )
        
        trainer = HFSegmentationTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            num_classes=len(train_dataset.get_class_map())-1, 
            ignore_index=0,
            dice_loss_kwargs={"from_logits":True},
            compute_metrics=lambda eval_pred: HFSegmentationTrainer.compute_metrics(trainer, eval_pred)
        )
            
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total training time: {total_time / 60:.2f} minutes")
        
        valid_metrics = trainer.evaluate(eval_dataset=valid_dataset)
        print(valid_metrics)
        
        test_metrics = save_metrics(trainer, test_dataset, f"{self.model_name.capitalize()} - {self.model_size.capitalize()}", os.path.join(output_path, "metrics.csv"), mode="test", training_time=total_time)

        return {
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "training_time_min": total_time,
    }
        
    



