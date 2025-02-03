from src.transformers.transformer import Transformer

from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

class MultilabelToYOLOTransformer(Transformer):

    def __init__(self):
        super().__init__()

    def transform(self, input_data: str, output_dir: str, num_classes: int):
        convert_segment_masks_to_yolo_seg(masks_dir=input_data, output_dir=output_dir, classes=num_classes)
