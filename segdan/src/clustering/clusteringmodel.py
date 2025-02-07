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


    def _clustering(self, scoring_function):
        if self.plot:
            labels = self.model.clustering(**self.args, output=self.output_path)
        else:
            labels = self.model.clustering(**self.args)
        return scoring_function(self.embeddings, labels)

    def train(self, model_name, verbose: bool):

        print(f"Using {model_name} model for clustering image embeddings...")
        
        if not any("_range" in key for key in self.args):
            scoring_function = get_scoring_function(self.metric)

            clustering_args = {
                'reduction': self.visualization_technique
            }

            if model_name == 'kmeans':
                clustering_args.update({
                'n_clusters': self.args["n_clusters"]
                })
                score = self._clustering(scoring_function)
                return clustering_args['n_clusters'], score
            
            elif model_name == 'agglomerative':
                clustering_args.update({
                'n_clusters': self.args["n_clusters"],
                'linkage': self.args["linkage"]
                })
                score = self._clustering(scoring_function)
                return clustering_args['n_clusters'], clustering_args['linkage'], score
            
            elif model_name == 'dbscan':
                clustering_args.update({
                'eps': self.args["eps"],
                'min_samples': self.args["min_samples"]
                })
                score = self._clustering(scoring_function)
                return clustering_args['eps'], clustering_args['min_samples'], score
            
            elif model_name == 'optics':
                clustering_args.update({
                'min_samples': self.args["min_samples"]
                })
                score = self._clustering(scoring_function)
                return clustering_args['min_samples'], score

        if model_name == 'kmeans':
            return self.model.find_best_n_clusters(self.args["n_clusters_range"], self.metric, self.plot, self.output_path)
        if model_name == 'agglomerative':
            return self.model.find_best_agglomerative_clustering(self.args["n_clusters_range"], self.metric, self.args["linkages"], self.plot, self.output_path)
        if model_name == 'dbscan':
            return self.model.find_best_DBSCAN(self.args["eps_range"], self.args["min_samples_range"], self.metric, self.plot, self.output_path, verbose)
        if model_name == 'optics':
            return self.model.find_best_OPTICS(self.args["min_samples_range"], self.metric, self.plot, self.output_path, verbose)