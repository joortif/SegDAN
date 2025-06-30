from segdan.utils.constants import LabelFormat
from segdan.converters.converterfactory import ConverterFactory
from segdan.utils.imagelabelutils import ImageLabelUtils

from imagedatasetanalyzer import ImageLabelDataset, ImageDataset
import os 

def convert_to_mask(label_path, image_path, input_format, general_data, output_path, converter_factory):
    transformations_path = os.path.join(output_path, "transformations", LabelFormat.MASK.value)
    os.makedirs(transformations_path, exist_ok=True)

    print(f"Transforming labels from {input_format} to multilabel. Results are saved in {transformations_path}")

    context = {
        "input_data": label_path,
        "img_dir": image_path,
        "output_dir": transformations_path,
        **general_data  
    }

    converter = converter_factory.get_converter(input_format, LabelFormat.MASK.value, context)
    converter.convert()

    return transformations_path

def _analyze_and_save_results(dataset: ImageDataset, output_path: str, verbose: bool):
    analysis_result_path = os.path.join(output_path, "analysis")
    os.makedirs(analysis_result_path, exist_ok=True)

    print("Starting dataset analysis...")
    dataset.analyze(output=analysis_result_path, verbose=verbose)

    print(f"Dataset analysis ended successfully. Results saved in {analysis_result_path}")

def analyze_data(general_data: dict, transformerFactory: ConverterFactory, output_path:str, class_map: dict,  verbose: bool):

    image_path = general_data["image_path"]
    label_path = general_data["label_path"]
    label_format = general_data["label_format"]
    background = general_data.get("background", None)
    binary = general_data.get("binary", False)

    if label_path is None:
        dataset = ImageDataset(image_path)
        _analyze_and_save_results(dataset, output_path, verbose)
        return 

    if label_format == LabelFormat.MASK.value:
        if binary:
            label_format = LabelFormat.BINARY.value
        elif ImageLabelUtils.all_images_are_color(label_path):
            label_format = LabelFormat.COLOR.value
        
    if label_format != LabelFormat.MASK.value:
        label_path = convert_to_mask(label_path, image_path, label_format, general_data, output_path, transformerFactory)

    dataset = ImageLabelDataset(image_path, label_path, background=background, class_map=class_map)
    _analyze_and_save_results(dataset, output_path, verbose)
    return 
        

