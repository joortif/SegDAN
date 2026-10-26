from tempfile import TemporaryDirectory
from typing import List, Optional
from pathlib import Path

import numpy as np
import cv2

from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

from segdan.extensions.extensions import LabelExtensions
from segdan.converters.converter import Converter

class MultilabelToYOLOConverter(Converter):

    def __init__(self, input_data: str, output_dir: str, num_classes: int, background: Optional[int] = 0, class_values: Optional[List[int]] = None):
        super().__init__(input_data, output_dir)
        self.num_classes = num_classes
        self.background = 0 if background is None else background
        self.class_values = ([int(value) for value in class_values] if class_values is not None else None)

    def convert(self):
        input_dir = Path(self.input_data)
        output_dir = Path(self.output_dir)

        mask_paths = sorted(p for p in input_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in {LabelExtensions.PNG.value, LabelExtensions.JPG.value, LabelExtensions.JPEG.value, 
                                                                    LabelExtensions.TIFF.value, LabelExtensions.BMP.value})

        masks = {}
        observed_values = set()

        for path in mask_paths:
            mask = cv2.imread(
                str(path), cv2.IMREAD_UNCHANGED
            )

            values = {int(value) for value in np.unique(mask)}
            observed_values.update(values)
            masks[str(path)] = mask
            
        source_classes = self.class_values
        if source_classes is None:
            source_classes = sorted(observed_values - {self.background})

        if len(source_classes) != self.num_classes:
            raise ValueError(f"Number of source classes ({len(source_classes)}) does not match the expected number of classes ({self.num_classes}).")

        if len(set(source_classes)) != len(source_classes):
            raise ValueError("Source classes contain duplicate values.")

        if self.background in source_classes:
            raise ValueError(f"Background value ({self.background}) should not be included in source classes.")

        unknown = (
            observed_values
            - set(source_classes)
            - {self.background}
        )

        if unknown:
            raise ValueError(
                f"Unknown class values found in the masks: {unknown}. "
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        with TemporaryDirectory() as tmp:
            normalized_dir = Path(tmp)

            for filename, mask in masks.items():
                normalized = np.zeros(mask.shape, dtype=np.uint8)

                for target_value, source_value in enumerate(
                    source_classes, start=1
                ):
                    normalized[mask == source_value] = target_value

                cv2.imwrite(
                    str(normalized_dir / filename),
                    normalized,
                )

            convert_segment_masks_to_yolo_seg(
                masks_dir=str(normalized_dir),
                output_dir=str(output_dir),
                classes=self.num_classes,
            )
