import os
from typing import Optional
import numpy as np
import torch
import re
import logging

from segdan.exceptions.exceptions import NoValidAutobatchConfigException
from segdan.training.autobatch import autobatch

logger = logging.getLogger(__name__)

class SemanticSegmentationModel:

    def __init__(self, classes: np.ndarray, epochs:int, imgsz:int, metrics: np.ndarray, selection_metric: str, model_name:str, model_size:str, output_path:str, fraction:Optional[float]=0.6):
        
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

        self.lr = 2e-4

    def save_model(self, output_dir, weights_only=True):
        model_save_name = f"{self.model_name}-{self.model_size}-ep{self.epochs}.pth"
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
        
    def run_training():
        raise NotImplementedError("Subclasses must implement this method") 

    def save_metrics():
        raise NotImplementedError("Subclasses must implement this method")