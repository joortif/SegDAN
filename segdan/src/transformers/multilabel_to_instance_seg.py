from src.transformers.transformer import Transformer

import numpy as np
from skimage.measure import label, regionprops

class MultilabelToInstanceSegmentationTransformer(Transformer):

    def __init__(self):
        super().__init__()


    def transform(self, input_data: np.ndarray, background: int | None = None ) -> np.ndarray:

        instance_masks = []

        for mask in input_data:
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
        
        return np.array(instance_masks)