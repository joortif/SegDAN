import os
import shutil
import json
import logging

from segdan.utils.constants import ReductionMethods
from segdan.extensions.extensions import LabelExtensions
from segdan.utils.imagelabelutils import ImageLabelUtils
from segdan.reduction.reduction_strategies import fast_vote_K, reduce_clusters

logger = logging.getLogger(__name__)

def save_labels_subset(image_dir, image_files, labels_dir, label_extension, output_path):

    labels = []

    for img_file in image_files:
        
        img_path = os.path.join(image_dir, img_file)
        label = ImageLabelUtils.image_to_label(img_path, labels_dir, label_extension)
        
        shutil.copy(label, output_path)

    return labels

def reduce_JSON(file, image_files, output_path):
    with open(file) as f:
        data = json.load(f)

    image_id_map = {img['id']: img for img in data['images'] if img['file_name'] in image_files}
    filtered_annotations = [ann for ann in data['annotations'] if ann['image_id'] in image_id_map]

    reduced_data = {
        "images": list(image_id_map.values()),
        "annotations": filtered_annotations,
        "categories": data.get("categories", [])  
    }
    
    with open(output_path, 'w') as f:
        json.dump(reduced_data, f, indent=4)

def reduce_dataset(config, clustering_results, evaluation_metric, dataset, label_path, embeddings_dict, output_path, verbose):
    retention_percentage = config.get('retention_percentage')
    use_reduced = config.get('use_reduced')
    method = config.get('reduction_type')

    output_path = os.path.join(output_path, "reduction", "images" if use_reduced else "")
    os.makedirs(output_path, exist_ok=True)
    
    if method.lower() == ReductionMethods.VOTE_K.value:
        reduced_ds = fast_vote_K(embeddings_dict=embeddings_dict, retention_percentage=retention_percentage, dataset=dataset, output_dir=output_path)
    else:
        reduced_ds = reduce_clusters(config=config, dataset=dataset, embeddings_dict=embeddings_dict, retention_percentage=retention_percentage, 
                                     clustering_results=clustering_results, evaluation_metric=evaluation_metric, output_path=output_path,
                                     verbose=verbose)

    if use_reduced and label_path:
        label_extension = ImageLabelUtils.check_label_extensions(label_path)
                
        label_output_path = os.path.join(os.path.dirname(output_path), "labels")
        os.makedirs(label_output_path, exist_ok=True)
        labels_dir = label_path if os.path.isdir(label_path) else os.path.dirname(label_path)

        if label_extension == LabelExtensions.enumToExtension(LabelExtensions.JSON):
            
            output_file_path = os.path.join(label_output_path, "reduced_annotations.json")
            reduce_JSON(os.path.join(labels_dir, label_path), reduced_ds.image_files, output_file_path)
        else:   
            save_labels_subset(output_path, reduced_ds.image_files, labels_dir, label_extension, label_output_path)

    return os.path.dirname(output_path)