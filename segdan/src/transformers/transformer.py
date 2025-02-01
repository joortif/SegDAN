
import numpy as np
import cv2
import logging
from transformers.utils import logging as hf_logging
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch

class Transformer():

    def __init__(self):
        
        hf_logging.set_verbosity_error()

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers(): 
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _create_empty_mask(self, height: int, width: int, fill_background: int |None):

        if fill_background is None:
          fill_background = 0
                
        return np.full((height, width), fill_background, dtype=np.uint8)
    
    def _scale_polygon(self, polygon, height: int, width: int):
        polygon[:, 0] *= width
        polygon[:, 1] *= height

        return polygon
    
    def _load_depth_model(self, model_name, device, verbose):
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)

        if verbose:
            self.logger.info(f"Model {model_name} loaded.")

        return model, processor
    
    def _generate_depth_map(self, image, model, processor, device):
        
        img = cv2.imread(image)
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        inputs = processor(images=img_rgb, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            depth_map = outputs.predicted_depth

        depth_map = depth_map.squeeze().cpu().numpy()

        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth_map.astype(np.uint8)



    def transform(self, input_data):
        pass