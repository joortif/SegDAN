from src.transformers.transformer import Transformer
from src.utils.imagelabelutils import ImageLabelUtils

import numpy as np
import os
import cv2
from tqdm import tqdm

class BinaryToMultilabelTransformer(Transformer):

    def __init__(self):
        super().__init__()

    def transform(self, input_data: str, output_dir: str, threshold: int = 255):
        masks = []

        label_files = [f for f in os.listdir(input_data) if os.path.isfile(os.path.join(input_data, f))]

        os.makedirs(output_dir, exist_ok=True)

        for filename in tqdm(label_files, desc="Converting labels from binary format to multilabel..."):
            label_path = os.path.join(input_data, filename)

            mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            converted_mask = (mask >= threshold).astype(np.uint8)  

            ImageLabelUtils.save_multilabel_mask(converted_mask, filename, output_dir)
            masks.append(converted_mask)
            
        return masks

