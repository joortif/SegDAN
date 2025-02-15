from src.clustering.clusteringmodel import ClusteringModel
from src.clustering.clusteringfactory import ClusteringFactory
from src.clustering.embeddingfactory import EmbeddingFactory

from imagedatasetanalyzer import ImageDataset
import os

def get_embeddings(config_data:dict, dataset:ImageDataset, verbose:bool, logger = None):
    
    emb_factory = EmbeddingFactory()

    config_analysis = config_data['analysis']

    emb_model = emb_factory.get_embedding_model(config_data, config_analysis)

    embedding_info = config_analysis.get("embedding_model")

    if verbose:
        logger.info(f"Successfully loaded {embedding_info.get('name') or 'LBP'} model from {embedding_info.get('framework')}.")

    embeddings = emb_model.generate_embeddings(dataset)

    return embeddings


def cluster_images(config_data: dict, dataset: ImageDataset, embeddings, output_path, verbose: bool, logger = None):

    clustering_factory = ClusteringFactory()

    plot = config_data['plot']
    evaluation_metric = config_data['clustering_metric']
    vis_technique = config_data['visualization_technique']
    
    clustering_models = config_data["clustering_models"]

    results = {}
    for (model_name, args) in clustering_models.items():
        random_state = args.get("random_state", 123)
        clust_model = clustering_factory.generate_clustering_model(model_name, dataset, embeddings, random_state)
        if plot:
            output_dir = os.path.join(output_path, "clustering", model_name)
            os.makedirs(output_dir, exist_ok=True)
        model = ClusteringModel(clust_model, args, embeddings, evaluation_metric,vis_technique, plot, output_dir)
        results[model_name] = model.train(model_name, verbose)

    if len(results.keys()) == 1:
        return results

    if evaluation_metric == 'davies':
        best_model = min(results.items(), key=lambda item: item[1][-1])
    else:
        best_model = max(results.items(), key=lambda item: item[1][-1])

    model_name = best_model[0]
    model_score = best_model[1][-1]  
    best_model_config = {
        'model_name': model_name,
        'score': model_score
    }

    model_params = clustering_models.get(model_name, {})

    logger.info(f"Best model: {model_name}")
    logger.info(f"Score ({evaluation_metric}): {model_score}")
    logger.info("Best parameter:")

    for param, value in model_params.items():
        best_value = best_model[1][list(model_params.keys()).index(param)]

        if '_range' in param:
            param = param.replace('_range', '')

        if param == 'random_state':
            best_value = value

        logger.info(f"  {param}: {best_value}")
        best_model_config[param] = best_value

    return best_model_config
