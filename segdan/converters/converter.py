from typing import Optional
import numpy as np
import logging
from transformers.utils import logging as hf_logging

logger = logging.getLogger(__name__)

class Converter():

    def __init__(self, input_data: str, output_dir: str):

        self.input_data = input_data
        self.output_dir = output_dir
        
        hf_logging.set_verbosity_error()

    def _create_empty_mask(self, height: int, width: int, fill_background: Optional[int]):

        if fill_background is None:
          fill_background = 0
                
        return np.full((height, width), fill_background, dtype=np.uint8)
    
    def _scale_polygon(self, polygon, height: int, width: int):
        polygon[:, 0] *= width
        polygon[:, 1] *= height

        return polygon
    
    
    def convert(self):
        raise NotImplementedError("Subclasses must implement this method")
