import numpy as np

from segdan.src.transformers.transformer import Transformer

class ColorToMultilabelTransformer(Transformer):

    def __init__(self):
        super().__init__()
    
    def transform(self, input_data: np.ndarray, color_dict: dict) -> np.ndarray:
        masks = []

        for color_mask in input_data:
            multilabel_mask = np.zeros(color_mask.shape[:2], dtype=np.uint8)

            pixel_colors = color_mask.reshape(-1, color_mask.shape[-1])

            for idx, pixel_color in enumerate(pixel_colors):
                color_tuple = tuple(pixel_color)
                if color_tuple in color_dict:
                    multilabel_mask.reshape(-1)[idx] = color_dict[color_tuple]

            multilabel_mask = multilabel_mask.reshape(color_mask.shape[:2])

            masks.append(multilabel_mask)
    
        return masks