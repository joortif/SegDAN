
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

def _find_best_model(clustering_model_configurations, evaluation_metric, logger, verbose):
 
    if evaluation_metric == 'davies':
        best_model = min(clustering_model_configurations.items(), key=lambda item: item[1][-2])
    else:
        best_model = max(clustering_model_configurations.items(), key=lambda item: item[1][-2])

    model_name = best_model[0]
    model_score = best_model[1][-2]  
    best_model_config = {
        'model_name': model_name,
        'score': model_score
    }
    best_model_labels = best_model[1][-1]

    model_params = clustering_model_configurations.get(model_name, {})

    if verbose:
        logger.info(f"Best model: {model_name}")
        logger.info(f"Score ({evaluation_metric}): {model_score}")
        logger.info("Best parameters:")

        for param, value in model_params.items():
            best_value = best_model[1][list(model_params.keys()).index(param)]

            if '_range' in param:
                param = param.replace('_range', '')

            if param == 'random_state':
                best_value = value

            logger.info(f"  {param}: {best_value}")
            best_model_config[param] = best_value

    return best_model_config, best_model_labels

def reduce_dataset(config, clustering_results, evaluation_metric, dataset, label_path, embeddings, output_path, verbose, logger):
    reduction_percentage = config['reduction_percentage']
    diverse_percentage = config['diverse_percentage']
    include_outliers = config['include_outliers']
    reduction_type = config['reduction_type']
    use_reduced = config['use_reduced']
    reduction_model_name = config['reduction_model']

    if reduction_model_name == "best_model":
        reduction_model, labels = _find_best_model(clustering_results, evaluation_metric, logger, verbose)
        reduction_model_name = reduction_model["model_name"]
    else:
        reduction_model_info = clustering_results[reduction_model_name]
        labels = reduction_model_info[-1]

    print(f"Using {reduction_model_name} model for dataset reduction.")

    random_state = reduction_model_info[-1][1] if reduction_model_name == 'kmeans' else 123

    clustering_factory = ClusteringFactory()
    model = clustering_factory.generate_clustering_model(reduction_model_name, dataset, embeddings, random_state)

    output_dir = os.path.join(output_path, "reduction", "images" if use_reduced else "")
    os.makedirs(output_dir, exist_ok=True)

    select_params = {
        "reduction": reduction_percentage,
        "diverse_percentage": diverse_percentage,
        "selection_type": reduction_type,
        "existing_labels": labels,
        "output_directory": output_dir
    }

    if reduction_model_name == "kmeans":
        select_params.pop("existing_labels")
        select_params["n_clusters"] = reduction_model_info[-1][0]
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