import os
from typing import Optional
import numpy as np
import torch
import re
import logging
import pandas as pd

from segdan.exceptions.exceptions import NoValidAutobatchConfigException
from segdan.training.autobatch import autobatch
from segdan.utils.utils import Utils

logger = logging.getLogger(__name__)

class SemanticSegmentationModel:

    def __init__(self, classes: np.ndarray, epochs:int, imgsz:int, metrics: np.ndarray, selection_metric: str, model_name:str, 
                 model_size:str, output_path:str, val_fold: Optional[int] = None, fraction:Optional[float]=0.6):
        
        self.classes = classes
        self.out_classes = len([cls for cls in self.classes if cls.lower() !="background"])
        self.epochs = epochs
        self.imgsz = imgsz
        self.metrics = metrics
        self.selection_metric = selection_metric
        self.model_name = model_name
        self.model_size = model_size
        self.output_path = output_path
        self.fraction = fraction

        if val_fold:
            self.val_fold=str(val_fold)

        self.lr = 2e-4

    def save_model(self, output_dir, weights_only=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        model_save_name = f"{self.model_name}-{self.model_size}-ep{self.epochs}"
        if self.val_fold is not None:
            model_save_name += f"-fold{self.val_fold}"
        model_save_name += ".pt"
        
        output_path = os.path.join(output_dir,model_save_name)

        if weights_only:
            torch.save(self.model.state_dict(), output_path)
            # logger.info(f"Model weights saved in {output_path}")
        else:
            torch.save(self.model, output_path)
            # logger.info(f"Complete model saved in {output_dir}")
        
        return output_path
    
    def show_metrics(self, metrics, stage):
    
        logger.info(f"{stage} metrics:\n")

        general_metrics = {k: v for k, v in metrics.items() if '_class_' not in k}

        logger.info(f"{'Metric':<20} {'Value':>8}")
        logger.info("-" * 30)
        for metric, value in general_metrics.items():
            logger.info(f"{metric:<20} {value:>8.4f}")

        logger.info("\nMetrics by Class:\n")

        class_pattern = re.compile(r'(.+)_class_(.+)')

        class_metrics = {}

        for key, value in metrics.items():
            match = class_pattern.match(key)
            if match:
                metric_name = match.group(1)
                class_idx = match.group(2)
                class_metrics.setdefault(metric_name, {})[class_idx] = value

        all_classes = sorted(set(idx for metric_dict in class_metrics.values() for idx in metric_dict))

        for metric_name, class_dict in class_metrics.items():
            logger.info(f"{metric_name.replace('_', ' ').title()}:")
            logger.info("-" * 30)
            for c in all_classes:
                val = class_dict.get(c, None)
                if val is not None:
                    logger.info(f"Class {c:<2} : {val:>8.4f}")
                else:
                    logger.info(f"Class {c:<2} : {'N/A':>8}")
            logger.info()

    def autobatch_imgsz(self):
        device = next(self.model.parameters()).device

        if device.type == "cpu" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
            
        try:
            self.batch = autobatch(model=self.model, imgsz=self.imgsz, fraction=self.fraction)
        except NoValidAutobatchConfigException as e:
            logger.info(f"Autobatch failed: {e}")
        
        if self.batch < 16:
            self.lr = 2e-5
            logger.info(f"Reducing learning rate to {self.lr}")

    def save_metrics(self, metrics, experiment_name, filename, hf=False, training_time=None):
        if not metrics:
            print("No metrics to save.")
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
            df.to_csv(file_output_path, sep=';', decimal=",", mode='a', header=False, index=False)
        else:
            df.to_csv(file_output_path, sep=';', decimal=",",  index=False)  

        print(f"Metrics saved in file {file_output_path}")

        evaluation_metric = metrics_dict.get(f"{self.selection_metric}_test")
        return evaluation_metric
        
    def run_training():
        raise NotImplementedError("Subclasses must implement this method") 

    def save_metrics():
        raise NotImplementedError("Subclasses must implement this method")