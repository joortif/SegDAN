from utils.constants import ClusteringModelName
from extensions.extensions import LabelExtensions
from utils.imagelabelutils import ImageLabelUtils
from clustering.clusteringfactory import ClusteringFactory
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

def _find_best_model(clustering_model_configurations, evaluation_metric, logger, verbose):
 
    if evaluation_metric == 'davies':
        best_model = min(clustering_model_configurations.items(), key=lambda item: item[1][-2])
    else:
        best_model = max(clustering_model_configurations.items(), key=lambda item: item[1][-2])

    model_name = best_model[0]
    model_values = best_model[1]
    model_score = model_values[-2]  
    best_model_labels = model_values[-1]

    param_names_dict = {
        "kmeans": ["n_clusters", "random_state"],
        "agglomerative": ["n_clusters", "linkage"],
        "dbscan": ["eps", "min_samples"],
        "optics": ["min_samples"]
    }

    param_names = param_names_dict.get(model_name, [])
    param_values = model_values[:len(param_names)]
    model_params = dict(zip(param_names, param_values))

    best_model_config = {
        'model_name': model_name,
        'score': model_score,
        **model_params
    }
    
    if verbose:
        logger.info(f"Best model: {model_name}")
        logger.info(f"Score ({evaluation_metric}): {model_score}")
        logger.info("Best parameters:")

        for param, value in model_params.items():
            logger.info(f"  {param}: {value}")

    return best_model_config, best_model_labels

def reduce_dataset(config, clustering_results, evaluation_metric, dataset, label_path, embeddings, output_path, verbose, logger):
    reduction_percentage = config['reduction_percentage']
    diverse_percentage = config['diverse_percentage']
    include_outliers = config['include_outliers']
    reduction_type = config['reduction_type']
    use_reduced = config['use_reduced']
    reduction_model_name = config['reduction_model']

    if reduction_model_name == "best_model":
        reduction_model_info, labels = _find_best_model(clustering_results, evaluation_metric, logger, verbose)
        reduction_model_name = reduction_model_info["model_name"]
    else:
        reduction_model_info = clustering_results[reduction_model_name]
        labels = reduction_model_info[-1]

    print(f"Using {reduction_model_name} model for dataset reduction.")

    random_state = reduction_model_info["random_state"] if reduction_model_name == 'kmeans' else 123

    clustering_factory = ClusteringFactory()
    model = clustering_factory.generate_clustering_model(reduction_model_name, dataset, embeddings, random_state)

    output_dir = os.path.join(output_path, "reduction", "images" if use_reduced else "")
    os.makedirs(output_dir, exist_ok=True)

    select_params = {
        "retention_percentage": reduction_percentage,
        "diverse_percentage": diverse_percentage,
        "selection_type": reduction_type,
        "existing_labels": labels,
        "output_directory": output_dir
    }

    if reduction_model_name == ClusteringModelName.KMEANS.value:
        select_params.pop("existing_labels")
        select_params["n_clusters"] = reduction_model_info["n_clusters"]
    elif reduction_model_name in ["dbscan", "optics"]:
        select_params["include_outliers"] = include_outliers

    reduced_ds = model.select_balanced_images(**select_params)

    if use_reduced and label_path:
        label_extension = ImageLabelUtils.check_label_extensions(label_path)
        
        if 'images' in output_dir.split(os.sep):
            image_path = output_dir
            output_dir = os.path.join(output_path, "reduction")
        
        label_output_path = os.path.join(output_dir, "labels")
        os.makedirs(label_output_path, exist_ok=True)
        labels_dir = label_path if os.path.isdir(label_path) else os.path.dirname(label_path)

        if label_extension == LabelExtensions.enumToExtension(LabelExtensions.JSON):
            
            output_file_path = os.path.join(label_output_path, "reduced_annotations.json")
            reduce_JSON(os.path.join(labels_dir, label_path), reduced_ds.image_files, output_file_path)
        else:   
            save_labels_subset(image_path, reduced_ds.image_files, labels_dir, label_extension, label_output_path)

    return os.path.join(output_path, "reduction")