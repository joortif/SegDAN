from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
from src.converters.converter import Converter

class MultilabelToYOLOConverter(Converter):

    def __init__(self, input_data: str, output_dir: str, num_classes: int):
        super().__init__(input_data, output_dir)
        self.num_classes = num_classes

    def convert(self):
        convert_segment_masks_to_yolo_seg(masks_dir=self.input_data, output_dir=self.output_dir, classes=self.num_classes)
