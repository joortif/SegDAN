from src.utils.utils import Utils
from src.metrics.clusteringmetrics import get_scoring_function

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
            labels = self.model.clustering(**params, output=self.output_path)
        else:
            labels = self.model.clustering(**params)
        return scoring_function(self.embeddings, labels), labels
    
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

        if model_name == "kmeans":
            return self.model.find_best_n_clusters(params["n_clusters_range"], self.metric, self.plot, self.output_path)
        if model_name == "agglomerative":
            return self.model.find_best_agglomerative_clustering(
                params["n_clusters_range"], self.metric, params["linkages"], self.plot, self.output_path
            )
        if model_name == "dbscan":
            return self.model.find_best_DBSCAN(
                params["eps_range"], params["min_samples_range"], self.metric, self.plot, self.output_path, verbose
            )
        if model_name == "optics":
            return self.model.find_best_OPTICS(
                params["min_samples_range"], self.metric, self.plot, self.output_path, verbose
        )