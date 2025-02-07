
from src.clustering.clusteringfactory import ClusteringFactory
import os

def reduce_dataset(config, best_model_config, dataset, embeddings):

    clustering_factory = ClusteringFactory()
    reduction_percentage = config['reduction_percentage']
    diverse_percentage = config['diverse_percentage']
    include_outliers = config['include_outliers']
    reduction_type = config['reduction_type']

    output_path = os.path.join(config['output_path'], "reduction")

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

    if model_name == 'kmeans' or model_name == 'dbscan':
        reduced_ds = model.select_balanced_images(**reduction_model_info, reduction=reduction_percentage, diverse_percentage=diverse_percentage, 
                                              selection_type=reduction_type, include_outliers=include_outliers, output_directory=output_path)
        return reduced_ds
    else:
        reduced_ds = model.select_balanced_images(**reduction_model_info, reduction=reduction_percentage, diverse_percentage=diverse_percentage, 
                                              selection_type=reduction_type, output_directory=output_path)
        return reduced_ds
    
    
