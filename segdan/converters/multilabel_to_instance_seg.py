import numpy as np
from skimage.measure import label, regionprops
import os
import cv2

from segdan.converters.converter import Converter


class MultilabelToInstanceSegmentationConverter(Converter):

    def __init__(self, input_data: str, output_dir: str, background: int | None = None ):
        super().__init__(input_data, output_dir)
        self.background = background


    def convert(self) -> np.ndarray:

        instance_masks = []

        label_files = [f for f in os.listdir(self.input_data) if os.path.isfile(os.path.join(self.input_data, f))]

        os.makedirs(self.output_dir, exist_ok=True)

        for filename in label_files:
            label_path = os.path.join(self.input_data, filename)

            mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            classes = np.unique(mask)

            if self.background is not None:
                classes = [cl for cl in classes if cl != self.background]

            for cl in classes:
                bin_mask = (mask == cl).astype(np.uint8)

                labeled_mask = label(bin_mask, connectivity=2)

                for region in regionprops(labeled_mask):

                    instance_mask = np.zeros_like(bin_mask)
                    instance_mask[labeled_mask == region.label] = cl

                    instance_masks.append(instance_mask)
        
        return instance_masks