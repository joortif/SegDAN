import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from imagedatasetanalyzer import ImageDataset
from tqdm import tqdm
import shutil

from segdan.clustering.clusteringfactory import ClusteringFactory
from segdan.utils.constants import ClusteringModelName

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

def reduce_clusters(config, dataset, embeddings_dict, retention_percentage, clustering_results, evaluation_metric, logger, output_path, verbose):
    diverse_percentage = config.get('diverse_percentage')
    include_outliers = config.get('include_outliers')
    reduction_type = config.get('reduction_type')
    use_reduced = config.get('use_reduced')
    reduction_model_name = config.get('reduction_model')
    embeddings = list(embeddings_dict.values())

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
        "retention_percentage": retention_percentage,
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
    return reduced_ds

def fast_vote_K(embeddings_dict:dict, retention_percentage: float, dataset: ImageDataset, output_dir: str, neighbours: int=150, p_factor: float=10.0):        

    embeddings = list(embeddings_dict.values())
    filenames = list(embeddings_dict.keys())
    
    total = len(embeddings)
    num_imgs_to_select = int(np.floor(retention_percentage * total))
    imgs_to_label = set()
    all_indices = set(range(total))
    unselected = all_indices.copy()

    graph = NearestNeighbors(n_neighbors=neighbours+1, metric='cosine').fit(embeddings)
    _, indices = graph.kneighbors(embeddings)
    knn_graph = {i: set(indices[i][1:]) for i in range(total)}

    labelled_neighbors_count = np.zeros(total, dtype=int)
    scores = np.zeros(total)

    for i in range(total):
        scores[i] = sum(p_factor ** -labelled_neighbors_count[v] for v in knn_graph[i])

    pbar = tqdm(total=num_imgs_to_select, desc="Selecting subset using Fast Vote-K")

    while len(imgs_to_label) < num_imgs_to_select:
        best_score_idx = max(unselected, key=lambda x: scores[x])

        imgs_to_label.add(best_score_idx)
        unselected.remove(best_score_idx)

        for neighbour in knn_graph[best_score_idx]:
            labelled_neighbors_count[neighbour] +=1
            if neighbour in unselected:
                scores[neighbour] = sum(p_factor ** -labelled_neighbors_count[v] for v in knn_graph[neighbour] if v in unselected)

        pbar.update(1) 

    pbar.close()

    selected_filenames = [filenames[i] for i in imgs_to_label]
    reduced_ds = ImageDataset(output_dir, selected_filenames)

    for filename in selected_filenames:
        src_path = os.path.join(dataset.img_dir, filename)
        dst_path = os.path.join(reduced_ds.img_dir, filename)
        shutil.copy(src_path, dst_path)

    return reduced_ds   
