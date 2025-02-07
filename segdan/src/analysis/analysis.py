from src.transformers.transformerfactory import TransformerFactory
from src.utils.imagelabelutils import ImageLabelUtils
from src.utils.confighandler import ConfigHandler
from src.extensions.extensions import LabelExtensions

from imagedatasetanalyzer import ImageLabelDataset, ImageDataset
import os 


def _transform_labels(labels_dir, imgs_dir, input, data, transformerFactory): 
    transformations_path = os.path.join(data["output_path"], "transformations", "multilabel")
        
    os.makedirs(transformations_path, exist_ok=True)

    print(f"Transforming labels from {input} to multilabel for the analysis. Results are saved in {transformations_path}")

    transformer = transformerFactory.get_converter(input, 'multilabel')

    if input in [LabelExtensions.TXT.value, LabelExtensions.JSON.value]:  
        transformer.transform(input_data=labels_dir, img_dir=imgs_dir, fill_background=data["background"], output_dir=transformations_path)
    
    elif input == "color":
        transformer.transform(input_data=labels_dir, output_dir=transformations_path, color_dict=data["color_dict"])

    elif input == "binary":
        transformer.transform(input_data=labels_dir, output_dir=transformations_path, threshold=data["threshold"])

    return transformations_path

def _analyze_and_save_results(dataset: ImageDataset, output_path: str, verbose: bool):
    analysis_result_path = os.path.join(output_path, "analysis")
    os.makedirs(analysis_result_path, exist_ok=True)

    dataset.analyze(output=analysis_result_path, verbose=verbose)

    print(f"Dataset analysis ended successfully. Results saved in {analysis_result_path}")

def analyze_data(data: dict, transformerFactory: TransformerFactory, logger):

    dataset_path = data["dataset_path"]
    output_path = data["output_path"]
    background = data.get("background", None)
    binary = data.get("binary")
    verbose = data.get("verbose", False)

    labels_dir = ConfigHandler.get_labels_dir(dataset_path)
    imgs_dir = os.path.join(dataset_path, "images")

    if labels_dir is None:
        dataset = ImageDataset(imgs_dir)
        _analyze_and_save_results(dataset, output_path, verbose)
        return 

    ext = ImageLabelUtils.check_label_extensions(labels_dir, verbose, logger)

    if ext != LabelExtensions.PNG.value:
        labels_dir = _transform_labels(labels_dir, imgs_dir, ext, data, transformerFactory)
    else: 
        if binary: 
            labels_dir = _transform_labels(labels_dir, imgs_dir, "binary", data, transformerFactory)
        if ImageLabelUtils.all_images_are_color(labels_dir):
            labels_dir = _transform_labels(labels_dir, imgs_dir, "color", data, transformerFactory)

    dataset = ImageLabelDataset(imgs_dir, labels_dir, background=background)
    _analyze_and_save_results(dataset, output_path, verbose)
    return 
        

    