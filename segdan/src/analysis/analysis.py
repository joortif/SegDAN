from src.transformers.transformerfactory import TransformerFactory
from src.transformers.transform import transform_labels
from src.utils.imagelabelutils import ImageLabelUtils
from src.utils.confighandler import ConfigHandler
from src.extensions.extensions import LabelExtensions

from imagedatasetanalyzer import ImageLabelDataset, ImageDataset
import os 

def _analyze_and_save_results(dataset: ImageDataset, output_path: str, verbose: bool):
    analysis_result_path = os.path.join(output_path, "analysis")
    os.makedirs(analysis_result_path, exist_ok=True)

    dataset.analyze(output=analysis_result_path, verbose=verbose)

    print(f"Dataset analysis ended successfully. Results saved in {analysis_result_path}")

def analyze_data(general_data: dict, transformerFactory: TransformerFactory, output_path:str,  verbose: bool, logger):

    image_path = general_data["image_path"]
    label_path = general_data["label_path"]
    background = general_data.get("background", None)
    binary = general_data.get("binary")

    if label_path is None:
        dataset = ImageDataset(image_path)
        _analyze_and_save_results(dataset, output_path, verbose)
        return 

    ext = ImageLabelUtils.check_label_extensions(label_path, verbose, logger)

    if ext != LabelExtensions.PNG.value:
        label_path = transform_labels(label_path, image_path, ext, general_data, output_path, background, transformerFactory)
    else: 
        if binary: 
            label_path = transform_labels(label_path, image_path, "binary", general_data, output_path, background, transformerFactory)
        if ImageLabelUtils.all_images_are_color(label_path):
            label_path = transform_labels(label_path, image_path, "color", general_data, output_path, background, transformerFactory)

    dataset = ImageLabelDataset(image_path, label_path, background=background)
    _analyze_and_save_results(dataset, output_path, verbose)
    return 
        

    