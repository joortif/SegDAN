import os
import shutil
from typing import Optional
import torch
import logging
import cv2
import matplotlib.pyplot as plt
import numpy as np
import segdan.utils.constants

logger = logging.getLogger(__name__)

class Utils():

    @staticmethod
    def get_device():

        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Device available in CUDA. Using GPU.")
        else:
            device = torch.device("cpu")
            logger.info("Could not find device in CUDA. Using CPU.")

        return device
    
    @staticmethod
    def params_to_range(range_params):
        start = range_params["min"]
        stop = range_params["max"]
        step = range_params["step"]
        
        if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
            return range(start, stop + 1, step)
        
        elif isinstance(start, (int, float)) and isinstance(stop, (int, float)) and isinstance(step, (int, float)):
            return np.arange(start, stop + step, step)
    
    def calculate_closest_resize(mode_height, mode_width, stride=32, max_padding=16):

        valid_sizes = [s for s in segdan.utils.constants.IMAGE_RESIZE_VALUES if s % stride == 0]

        original_mode_height = mode_height
        original_mode_width = mode_width

        if mode_height != mode_width:
            max_side = max(mode_height, mode_width)
            logger.info(f"Making size square by using {max_side}px height and {max_side}px width.")
            mode_height = mode_width = max_side

        closest_height = min(valid_sizes, key=lambda x: abs(x - mode_height))
        diff_height = abs(closest_height - mode_height)
        if diff_height <= max_padding:
            if closest_height != mode_height:
                logger.info(f"Dimensions {mode_height}px adjusted to {closest_height}px (compatible with stride {stride}).")
            mode_height = mode_width = closest_height
        else:
            lower_valid = [s for s in valid_sizes if s <= mode_height]
            new_height = max(lower_valid) if lower_valid else min(valid_sizes)
            if new_height != mode_height:
                logger.info(f"Dimensions {mode_height}px adjusted down to {new_height}px (compatible with stride {stride}).")
            mode_height = mode_width = new_height

        if original_mode_height != new_height or original_mode_width != new_height:
            logger.info(f"Images will be resized to {mode_height}px height and {mode_width}px width.")
            
        return mode_height
    
    def adapt_hf_metrics(metrics):
    
        #REMOVE COLUMNS
        columns_to_delete = ["epoch", "steps_per_second", "samples_per_second", "runtime", "loss"]

        keys_to_delete = [
            key for key in metrics.keys()
            if any(sub in key for sub in columns_to_delete)
        ]    
        
        for key in keys_to_delete:
            del metrics[key]
            
        #FORMAT NAMES
        new_metrics = {}
        for key, value in metrics.items():
            
            if key.startswith("eval_"):
                new_key = key[len("eval_"):]
            else:
                new_key = key
            
            new_metrics[new_key] = value
        
        new_metrics = {(key.replace("eval", "test")): value
                    for key, value in new_metrics.items()}
        
        return new_metrics
    
    def safe_remove(path: Optional[str]):
        if not path:
            return
        try:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
            else:
                os.remove(path)
        except Exception as e:
            logger.warning(f"Could not remove {path}: {e}")
