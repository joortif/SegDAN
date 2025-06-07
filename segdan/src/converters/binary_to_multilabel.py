from src.converters.converter import Converter
from src.utils.imagelabelutils import ImageLabelUtils

import numpy as np
import gc
import os
import cv2
from tqdm import tqdm

class BinaryToMultilabelConverter(Converter):

    def __init__(self, input_data: str, output_dir: str, threshold: int = 255):
        super().__init__(input_data, output_dir)
        self.threshold = threshold

    def convert(self):
        masks = []

        label_files = [f for f in os.listdir(self.input_data) if os.path.isfile(os.path.join(self.input_data, f))]

        os.makedirs(self.output_dir, exist_ok=True)

        for filename in tqdm(label_files, desc="Converting labels from binary format to multilabel..."):
            label_path = os.path.join(self.input_data, filename)
            
            mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            converted_mask = (mask >= self.threshold).astype(np.uint8)

            ImageLabelUtils.save_multilabel_mask(converted_mask, filename, self.output_dir)
            masks.append(converted_mask)

            del mask, converted_mask
            gc.collect()

        return masks

    


