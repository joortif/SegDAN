import os
import logging

from segdan.clustering.clusteringmodel import ClusteringModel
from segdan.clustering.clusteringfactory import ClusteringFactory
from segdan.clustering.embeddingfactory import EmbeddingFactory

from imagedatasetanalyzer import ImageDataset

logger = logging.getLogger(__name__)

def get_embeddings(clustering_data:dict, dataset:ImageDataset, verbose:bool):
    
    emb_factory = EmbeddingFactory()

    embedding_info = clustering_data.get("embedding_model")

    emb_model = emb_factory.get_embedding_model(embedding_info)

    if verbose:
        logger.info(f"Successfully loaded {embedding_info.get('name') or embedding_info.get('framework')} model.")

    embeddings = emb_model.generate_embeddings(dataset)

    return embeddings


def cluster_images(clustering_data: dict, dataset: ImageDataset, embeddings, output_path: str, verbose: bool, logger = None):

    clustering_factory = ClusteringFactory()

    plot = clustering_data['plot']
    evaluation_metric = clustering_data['clustering_metric']
    vis_technique = clustering_data['visualization_technique']
    
    clustering_models = clustering_data["clustering_models"]

    results = {}
    for (model_name, args) in clustering_models.items():
        random_state = args.get("random_state", 123)
        clust_model = clustering_factory.generate_clustering_model(model_name, dataset, embeddings, random_state)
        output_dir = None
        if plot:
            output_dir = os.path.join(output_path, "clustering", model_name)
            os.makedirs(output_dir, exist_ok=True)
        model = ClusteringModel(clust_model, args, embeddings, evaluation_metric,vis_technique, plot, output_dir)
        results[model_name] = model.train(model_name, verbose)

    return results