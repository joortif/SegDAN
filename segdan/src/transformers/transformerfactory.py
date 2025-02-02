from src.transformers.transformer import Transformer
from src.transformers.multilabel_to_yolo import MultilabelToYOLOTransformer
from src.transformers.multilabel_to_instance_seg import MultilabelToInstanceSegmentationTransformer
from src.transformers.color_to_multilabel import ColorToMultilabelTransformer
from src.transformers.json_to_multilabel import JSONToMultilabelTransformer
from src.transformers.yolo_to_multilabel import YOLOToMultilabelTransformer
from src.transformers.binary_to_multilabel import BinaryToMultilabelTransformer


class TransformerFactory:
    
    def __init__(self):
        self.converter_map = {
            ('binary', 'multilabel'): BinaryToMultilabelTransformer(),
            ('.txt', 'multilabel'): YOLOToMultilabelTransformer(),
            ('.json', 'multilabel'): JSONToMultilabelTransformer(),
            ('color', 'multilabel'): ColorToMultilabelTransformer(),
            ('multilabel','instance'): MultilabelToInstanceSegmentationTransformer(),
            ('multilabel', '.txt'): MultilabelToYOLOTransformer()
        }

    def get_converter(self, input_format: str, output_format: str) -> Transformer:
        try:
            return self.converter_map[(input_format, output_format)]
        except KeyError:
            raise ValueError(f'Transformation not supported for {input_format} format to {output_format} format.')