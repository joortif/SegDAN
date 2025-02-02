from src.transformers.transformer import Transformer

import numpy as np

class BinaryToMultilabelTransformer(Transformer):

    def __init__(self):
        super().__init__()

    def transform(self, input_data: np.ndarray, threshold: int = 255) -> np.ndarray:
        masks = []

        for mask in input_data: 
            converted_mask = (mask >= threshold).astype(np.uint8)  
            masks.append(converted_mask)

        return np.array(masks)

