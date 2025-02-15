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

def analyze_data(data: dict, transformerFactory: TransformerFactory, imgs_dir: str, verbose: bool, logger):

    analysis_data = data["analysis"]

    dataset_path = data["dataset_path"]
    output_path = data["output_path"]
    background = data.get("background", None)
    binary = data.get("binary")

    labels_dir = ConfigHandler.get_labels_dir(dataset_path)

    if labels_dir is None:
        dataset = ImageDataset(imgs_dir)
        _analyze_and_save_results(dataset, output_path, verbose)
        return 

    ext = ImageLabelUtils.check_label_extensions(labels_dir, verbose, logger)

    if ext != LabelExtensions.PNG.value:
        labels_dir = transform_labels(labels_dir, imgs_dir, ext, analysis_data, output_path, background, transformerFactory)
    else: 
        if binary: 
            labels_dir = transform_labels(labels_dir, imgs_dir, "binary", analysis_data, output_path, background, transformerFactory)
        if ImageLabelUtils.all_images_are_color(labels_dir):
            labels_dir = transform_labels(labels_dir, imgs_dir, "color", analysis_data, output_path, background, transformerFactory)

    dataset = ImageLabelDataset(imgs_dir, labels_dir, background=background)
    _analyze_and_save_results(dataset, output_path, verbose)
    return 
        

    