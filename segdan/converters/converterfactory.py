from dataclasses import dataclass
from typing import Iterable, Type

from segdan.converters.binary_to_yolo import BinaryToYOLOConverter
from segdan.utils.constants import LabelFormat
from segdan.converters.converter import Converter
from segdan.converters.binary_to_multilabel import BinaryToMultilabelConverter
from segdan.converters.color_to_multilabel import ColorToMultilabelConverter
from segdan.converters.json_to_multilabel import JSONToMultilabelConverter
from segdan.converters.multilabel_to_yolo import MultilabelToYOLOConverter
from segdan.converters.yolo_to_multilabel import YOLOToMultilabelConverter


@dataclass(frozen=True)
class ConverterSpec:
    """Metadata needed to instantiate a converter."""
    converter_class: Type[Converter]
    required_params: frozenset = frozenset()
    optional_params: frozenset = frozenset()


class ConverterFactory:

    def __init__(self):
        self.converter_map = {}

        # Binary mask -> multilabel mask
        self.register(
            LabelFormat.BINARY,
            LabelFormat.MASK,
            BinaryToMultilabelConverter,
            optional_params={"threshold"},
        )

        # YOLO TXT -> multilabel mask
        self.register(
            LabelFormat.TXT,
            LabelFormat.MASK,
            YOLOToMultilabelConverter,
            required_params={"img_dir", "background"},
            optional_params={"depth_model"},
        )

        # COCO JSON -> multilabel mask
        self.register(
            LabelFormat.JSON,
            LabelFormat.MASK,
            JSONToMultilabelConverter,
            required_params={"img_dir", "background"},
            optional_params={"depth_model"},
        )

        # Color mask -> multilabel mask
        self.register(
            LabelFormat.COLOR,
            LabelFormat.MASK,
            ColorToMultilabelConverter,
            optional_params={"color_dict"},
        )

        # Multilabel mask -> YOLO segmentation TXT
        self.register(
            LabelFormat.MASK,
            LabelFormat.TXT,
            MultilabelToYOLOConverter,
            required_params={"num_classes"},
            optional_params={"background", "class_values"},
        )

        self.register(
            LabelFormat.BINARY,
            LabelFormat.TXT,
            BinaryToYOLOConverter,
            required_params={"background", "foreground"}
        )

    @staticmethod
    def _normalize_format(label_format: str) -> str:
        value = getattr(label_format, "value", label_format)
        return str(value).lower()

    def register(
        self,
        input_format: str,
        output_format: str,
        converter_class: Type[Converter],
        required_params: Iterable[str] = (),
        optional_params: Iterable[str] = (),
    ):
        """Register a new input-format -> output-format transformation."""
        source = self._normalize_format(input_format)
        target = self._normalize_format(output_format)

        required = frozenset(required_params)
        optional = frozenset(optional_params)

        overlap = required & optional
        if overlap:
            raise ValueError(
                f"Parameters cannot be both required and optional: {overlap}"
            )

        key = (source, target)

        if key in self.converter_map:
            raise ValueError(f"Transformation already registered: {key}")

        self.converter_map[key] = ConverterSpec(
            converter_class=converter_class,
            required_params=required,
            optional_params=optional,
        )

    def get_converter(
        self,
        input_format: str,
        output_format: str,
        context: dict,
    ) -> Converter:

        source = self._normalize_format(input_format)
        target = self._normalize_format(output_format)
        key = (source, target)

        if key not in self.converter_map:
            raise ValueError(
                f"Unsupported transformation: {source} -> {target}"
            )

        spec = self.converter_map[key]

        base_params = {"input_data", "output_dir"}
        required = base_params | set(spec.required_params)

        missing = required - context.keys()
        if missing:
            raise ValueError(
                f"Missing required context parameters for "
                f"{spec.converter_class.__name__}: {sorted(missing)}"
            )

        init_kwargs = {
            name: context[name]
            for name in spec.required_params
        }

        for name in spec.optional_params:
            if name in context and context[name] is not None:
                init_kwargs[name] = context[name]

        return spec.converter_class(
            input_data=context["input_data"],
            output_dir=context["output_dir"],
            **init_kwargs,
        )

    def convert(
        self,
        input_format: str,
        output_format: str,
        context: dict,
    ):
        converter = self.get_converter(
            input_format, output_format, context
        )
        return converter.convert()

    def supports_conversion(
        self,
        input_format: str,
        output_format: str,
    ) -> bool:
        return (input_format, output_format) in self.converter_map