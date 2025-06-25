import numpy as np
import logging
from transformers.utils import logging as hf_logging

class Converter():

    def __init__(self, input_data: str, output_dir: str):

        self.input_data = input_data
        self.output_dir = output_dir
        
        hf_logging.set_verbosity_error()

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers(): 
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _create_empty_mask(self, height: int, width: int, fill_background: int | None):

        if fill_background is None:
          fill_background = 0
                
        return np.full((height, width), fill_background, dtype=np.uint8)
    
    def _scale_polygon(self, polygon, height: int, width: int):
        polygon[:, 0] *= width
        polygon[:, 1] *= height

        return polygon
    
    
    def convert(self):
        raise NotImplementedError("Subclasses must implement this method")
