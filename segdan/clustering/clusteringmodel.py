import numpy as np

from utils.utils import Utils
from metrics.clusteringmetrics import get_scoring_function
from utils.constants import ClusteringModelName

class ClusteringModel():

    def __init__(self, model, args, embeddings, metric, visualization_technique, plot, output_path):
        self.model = model
        self.args = args
        self.plot = plot
        self.output_path = output_path
        self.embeddings = embeddings
        self.metric = metric
        self.visualization_technique = visualization_technique


    def _clustering(self, scoring_function, params):
        if self.plot:
            labels = self.model.clustering(**params, reduction = self.visualization_technique, output=self.output_path)
        else:
            labels = self.model.clustering(**params)
        return scoring_function(self.embeddings, labels), labels
    
    def _plot_best_model(self, best_labels, model_name):
        num_clusters = len(np.unique(best_labels))
        embeddings_2d = self.model.reduce_dimensions(self.visualization_technique)
        self.model.plot_clusters(embeddings_2d, best_labels, num_clusters, self.visualization_technique, model_name, self.output_path)
    
    def get_param(self, param_name):
        
        if f"{param_name}_range" in self.args:
            return Utils.params_to_range(self.args[f"{param_name}_range"])
        return self.args.get(param_name)  
    
    def get_param(self, param_name):

        if "linkages" in self.args and param_name.strip() == "linkage":
            return self.args.get("linkages")

        if "_range" in param_name.strip():
            return Utils.params_to_range(self.args[f"{param_name}"])
        
        return self.args.get(param_name) 

    def train(self, model_name, verbose: bool):

        print(f"Using {model_name} model for clustering image embeddings...")

        scoring_function = get_scoring_function(self.metric)

        params = {param_name: self.get_param(param_name) for param_name in self.args.keys() if param_name != 'random_state'}

        if all(not "_range" in param for param in params):

            score, labels = self._clustering(scoring_function, params)

            if model_name == 'kmeans':

                params["random_state"] = self.args["random_state"]

            return (*params.values(), score, labels)

        if model_name == ClusteringModelName.KMEANS.value:
            best_n_clusters, random_state, best_score, best_labels =  self.model.find_best_n_clusters(params["n_clusters_range"], self.metric, self.plot, self.output_path)

            self._plot_best_model(best_labels, model_name)

            return best_n_clusters, random_state, best_score, best_labels
        
        if model_name == ClusteringModelName.AGGLOMERATIVE.value:

            best_k, best_linkage, best_score, best_labels = self.model.find_best_agglomerative_clustering(params["n_clusters_range"], self.metric, params["linkages"], self.plot, self.output_path)

            self._plot_best_model(best_labels, model_name)

            return best_k, best_linkage, best_score, best_labels
        
        if model_name == ClusteringModelName.DBSCAN.value:

            best_eps, best_min_samples, best_score, best_labels = self.model.find_best_DBSCAN(params["eps_range"], params["min_samples_range"], self.metric, self.plot, self.output_path, verbose)

            self._plot_best_model(best_labels, model_name)
            
            return best_eps, best_min_samples, best_score, best_labels
        if model_name == ClusteringModelName.OPTICS.value:

            best_min_samples, best_score, best_labels = self.model.find_best_OPTICS(params["min_samples_range"], self.metric, self.plot, self.output_path, verbose)

            self._plot_best_model(best_labels, model_name)

            return best_min_samples, best_score, best_labels