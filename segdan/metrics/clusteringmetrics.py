from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def get_scoring_function(metric: str):
    if metric == 'silhouette':
        scoring_function = silhouette_score
    elif metric == 'calinski':
        scoring_function = calinski_harabasz_score
    elif metric == 'davies':
        scoring_function = davies_bouldin_score
    else:
        raise ValueError(f"Unsupported metric: {metric}. Choose 'silhouette', 'calinski', or 'davies'.")
    
    return scoring_function