from src.utils.constants import LabelFormat
from src.converters.converter import Converter
from src.converters.binary_to_multilabel import BinaryToMultilabelConverter
from src.converters.color_to_multilabel import ColorToMultilabelConverter
from src.converters.json_to_multilabel import JSONToMultilabelConverter
from src.converters.multilabel_to_instance_seg import MultilabelToInstanceSegmentationConverter
from src.converters.multilabel_to_yolo import MultilabelToYOLOConverter
from src.converters.yolo_to_multilabel import YOLOToMultilabelConverter


class ConverterFactory:
    
    def __init__(self):
        self.converter_map = {
            (LabelFormat.BINARY.value, LabelFormat.MASK.value): (BinaryToMultilabelConverter, {'threshold'}),
            (LabelFormat.TXT.value, LabelFormat.MASK.value): (YOLOToMultilabelConverter, {'img_dir', 'background'}),
            (LabelFormat.JSON.value, LabelFormat.MASK.value): (JSONToMultilabelConverter, {'img_dir', 'background', 'depth_model'}),
            (LabelFormat.COLOR.value, LabelFormat.MASK.value): (ColorToMultilabelConverter, {'color_dict'}),
            (LabelFormat.MASK.value,LabelFormat.TXT.value): (MultilabelToYOLOConverter, {'num_classes'})
        }

    def get_converter(self, input_format: str, output_format: str, args: dict) -> Converter:
        try:
            converter_class, required_params = self.converter_map[(input_format, output_format)]
        except KeyError:
            raise ValueError(f"Unsupported transformation: {input_format} -> {output_format}")
        
        missing = required_params - args.keys()
        if missing:
            raise ValueError(f"Missing required context params: {missing} for converter {converter_class.__name__}")
        
        init_args = {key: args[key] for key in required_params if key in args}
        return converter_class(args["input_data"], args["output_dir"], **init_args)