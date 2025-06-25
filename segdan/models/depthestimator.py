from transformers import pipeline
from PIL import Image
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, depth_model="Intel/dpt-large", device=0):
        self.pipe = pipeline(task="depth-estimation", model=depth_model, device=device)

    def generate_depth_map(self, image_path):
        img = Image.open(image_path)

        result = self.pipe(img)
        depth_map = result["depth"]

        depth_map = np.array(depth_map)

        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = np.uint8(depth_map)

        w, h = img.size
        depth_map_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth_map_resized