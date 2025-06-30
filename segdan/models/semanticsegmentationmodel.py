import os
import numpy as np
import torch

class SemanticSegmentationModel:

    def __init__(self, out_classes: int, epochs:int, metrics: np.ndarray, selection_metric: str,
                 ignore_index:int, model_name:str, model_size:str, output_path:str):
        
        self.out_classes = out_classes
        self.epochs = epochs
        self.metrics = metrics
        self.selection_metric = selection_metric
        self.ignore_index = ignore_index
        self.model_name = model_name
        self.model_size = model_size
        self.output_path = output_path

    def save_model(self, output_dir, weights_only=True):
        model_save_name = f"{self.model_name}-{self.model_size}-ep{self.epochs}.pth"
        output_path = os.path.join(output_dir,model_save_name)

        if weights_only:
            torch.save(self.model.state_dict(), output_path)
            print(f"Model weights saved in {output_path}")
        else:
            torch.save(self.model, output_path)
            print(f"Complete model saved in {output_dir}")
        
        return output_path
        
    def train():
       raise NotImplementedError("Subclasses must implement this method") 

    def save_metrics():
        raise NotImplementedError("Subclasses must implement this method")