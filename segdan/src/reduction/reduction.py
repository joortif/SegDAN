
from src.extensions.extensions import LabelExtensions
from src.utils.imagelabelutils import ImageLabelUtils
from src.clustering.clusteringfactory import ClusteringFactory
import os
import shutil
import json

def save_labels_subset(image_dir, image_files, labels_dir, label_extension, output_path):

    labels = []

    for img_file in image_files:
        
        img_path = os.path.join(image_dir, img_file)
        label = ImageLabelUtils.image_to_label(img_path, labels_dir, label_extension)
        
        shutil.copy(label, output_path)

    return labels

def reduce_JSON(file, image_files, output_path):
    with open(file, 'r') as f:
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


def reduce_dataset(config, best_model_config, dataset, embeddings, dataset_path, output_path):

    clustering_factory = ClusteringFactory()

    reduction_percentage = config['reduction_percentage']
    diverse_percentage = config['diverse_percentage']
    include_outliers = config['include_outliers']
    reduction_type = config['reduction_type']
    use_reduced = config['use_reduced']

    clustering_models = config['clustering_models']
    random_state = 123
    if clustering_models.get('kmeans', None):
        random_state = clustering_models['kmeans'].pop('random_state')

    if config['reduction_model'] == 'best_model':
        model_name = best_model_config.pop('model_name')
        reduction_model_info = best_model_config

    else:
        reduction_model_info = next(iter(config['reduction_model'].values())) 
        model_name = next(iter(config['reduction_model']))  

    print(f"Using {model_name} for dataset reduction with params {reduction_model_info}")
    
    model = clustering_factory.generate_clustering_model(model_name, dataset, embeddings, random_state)

    output_dir = os.path.join(output_path, "reduction")
    
    if use_reduced:
        output_dir = os.path.join(output_dir, "images")

    if model_name == 'kmeans' or model_name == 'dbscan':
        reduced_ds = model.select_balanced_images(**reduction_model_info, reduction=reduction_percentage, diverse_percentage=diverse_percentage, 
                                              selection_type=reduction_type, include_outliers=include_outliers, output_directory=output_dir)
    else:
        reduced_ds = model.select_balanced_images(**reduction_model_info, reduction=reduction_percentage, diverse_percentage=diverse_percentage, 
                                              selection_type=reduction_type, output_directory=output_dir)
    
    if use_reduced:
        
        labels_dir = ImageLabelUtils.get_labels_dir(dataset_path)
        if labels_dir is None:
            return os.path.join(output_path, "reduction")
        
        label_extension = ImageLabelUtils.check_label_extensions(labels_dir)

        label_output_path = os.path.join(output_path, "reduction", "labels")
        os.makedirs(label_output_path, exist_ok=True)

        if label_extension == LabelExtensions.enumToExtension(LabelExtensions.JSON):

            json_file = os.listdir(labels_dir)[0]

            output_file_path = os.path.join(label_output_path, "reduced_annotations.json")

            reduce_JSON(os.path.join(labels_dir, json_file), reduced_ds.image_files, output_file_path)
        else:   
            save_labels_subset(output_dir, reduced_ds.image_files, labels_dir, label_extension, label_output_path)

        return os.path.join(output_path, "reduction")
    
    
