import numpy as np
import cv2
import os

from src.utils.imagelabelutils import ImageLabelUtils
from src.transformers.transformer import Transformer

class ColorToMultilabelTransformer(Transformer):

    def __init__(self):
        super().__init__()
    
    def transform(self, input_data: str, output_dir: str, color_dict: dict):
        masks = []
        os.makedirs(output_dir, exist_ok=True)

        label_files = [f for f in os.listdir(input_data) if os.path.isfile(os.path.join(input_data, f))]

        for filename in label_files:
            label_path = os.path.join(input_data, filename)
            color_mask = cv2.imread(label_path)
            multilabel_mask = np.zeros(color_mask.shape[:2], dtype=np.uint8)
            pixel_colors = color_mask.reshape(-1, color_mask.shape[-1])

            for idx, pixel_color in enumerate(pixel_colors):
                color_tuple = tuple(pixel_color)
                if color_tuple in color_dict:
                    multilabel_mask.reshape(-1)[idx] = color_dict[color_tuple]

            multilabel_mask = multilabel_mask.reshape(color_mask.shape[:2])
            masks.append(multilabel_mask)

            ImageLabelUtils.save_multilabel_mask(multilabel_mask, filename, output_dir)

        return masks