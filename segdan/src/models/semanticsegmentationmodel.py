import numpy as np

class SemanticSegmentationModel:

    def __init__(self, out_classes: int, epochs:int, metrics: np.ndarray, 
                 ignore_index:int, model_name:str, model_size:str):
        
        self.out_classes = out_classes
        self.epochs = epochs
        self.metrics = metrics
        self.ignore_index = ignore_index
        self.model_name = model_name
        self.model_size = model_size

    def save_weights():
        raise NotImplementedError("Subclasses must implement this method") 
    
    def save_model():
        raise NotImplementedError("Subclasses must implement this method") 

    def train():
       raise NotImplementedError("Subclasses must implement this method") 
