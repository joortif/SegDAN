from src.transformers.transformer import Transformer

import numpy as np
from skimage.measure import label, regionprops
import os
import cv2

class MultilabelToInstanceSegmentationTransformer(Transformer):

    def __init__(self):
        super().__init__()


    def transform(self, input_data: str, output_dir: str, background: int | None = None ) -> np.ndarray:

        instance_masks = []

        label_files = [f for f in os.listdir(input_data) if os.path.isfile(os.path.join(input_data, f))]

        os.makedirs(output_dir, exist_ok=True)

        for filename in label_files:
            label_path = os.path.join(input_data, filename)

            mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            classes = np.unique(mask)

            if background is not None:
                classes = [cl for cl in classes if cl != background]

            for cl in classes:
                bin_mask = (mask == cl).astype(np.uint8)

                labeled_mask = label(bin_mask, connectivity=2)

                for region in regionprops(labeled_mask):

                    instance_mask = np.zeros_like(bin_mask)
                    instance_mask[labeled_mask == region.label] = cl

                    instance_masks.append(instance_mask)
        
        return instance_masks