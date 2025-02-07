
from imagedatasetanalyzer import KMeansClustering, AgglomerativeClustering, DBSCANClustering, OPTICSClustering
from imagedatasetanalyzer import ImageDataset

import numpy as np

class ClusteringFactory():

    def generate_clustering_model(self, model_name: str, dataset: ImageDataset, embeddings: np.ndarray, random_state: int=123):
        
        models_map = {
            'kmeans': lambda: KMeansClustering(dataset, embeddings, random_state),
            'agglomerative': lambda: AgglomerativeClustering(dataset, embeddings, random_state),
            'dbscan': lambda: DBSCANClustering(dataset, embeddings, random_state),
            'optics': lambda: OPTICSClustering(dataset, embeddings, random_state)
        }

        if model_name not in models_map:
            raise ValueError(f"Clustering model '{model_name}' not supported. ")
        
        return models_map[model_name]()
    
    