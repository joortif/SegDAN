import json
import os
import yaml

class ConfigHandler():

    REQUIRED_KEYS = [
        "dataset_path", 
        "output_path", 
        "analyze",
        "cluster_images",
        "reduction",
        "split_percentages",
        "stratification",
        "segmentation", 
        "models", 
        "segmentation_metric", 
        "batch_size", 
        "epochs",
        "verbose"
        ]


    CONFIGURATION_VALUES = {
        "embedding_frameworks": ["huggingface", "pytorch", "tensorflow", "opencv"],
        "clustering_models": ["kmeans", "agglomerative", "dbscan", "optics"],
        "linkages" : ['ward', 'complete', 'average', 'single'],
        "visualization_techniques": ["pca", "tsne"],

        "lbp_methods": ["uniform", "default", "ror", "nri_uniform", "var"],

        "kmeans_clustering_metric": ["elbow", "silhouette", "calinski", "davies"],
        "clustering_metric": ["silhouette", "calinski", "davies"],
        "reduction_type": ["representative", "diverse", "random"],
        "reduction_models": ["best_model", "kmeans", "agglomerative", "dbscan", "optics"],

        "stratification_type": ["pixel_prop", "num_objects", "ratio"],

        "segmentation": ["semantic", "instance"],
        "semantic_segmentation_models": ["unet", "deeplabv3"],
        "instance_segmentation_models": ["yolo"],
        "segmentation_metric": ["iou", "dice_score"],
        "stratification_types": ["pixels", "objects", "pixel_to_object_ratio"]
    }

    DEFAULT_VALUES = {
        "resize_height": 224,
        "resize_width": 224,
        "lbp_radius": 8,
        "lbp_num_points": 24,
        "lbp_method": "uniform",
        "random_state": 123,
        "clustering_metric": "silhouette",
        "plot": True,
        "visualization_technique": "pca",
        "diverse_percentage": 0.0,
        "depth_model": "Intel/dpt-swinv2-tiny-256",
        "threshold": 255,
        "stratification_type": "pixels"
    }

    @staticmethod
    def get_labels_dir(dataset_path: str):
        try:
            labels_dir = os.path.join(dataset_path, "labels")
            if os.path.isdir(labels_dir):
                return labels_dir

            return None
        except Exception as e:
            return None
        
    @staticmethod
    def validate_range(range_params, param_name):
        if not isinstance(range_params, dict):
            raise ValueError(f"The '{param_name}' should be a dictionary with 'min', 'max', and 'step'.")
        
        if "min" not in range_params or "max" not in range_params or "step" not in range_params:
            raise ValueError(f"The '{param_name}' should contain 'min', 'max', and 'step' keys.")
        
        if not isinstance(range_params["min"], (int, float)):
            raise ValueError(f"'{param_name}.min' should be an integer or float.")
        if not isinstance(range_params["max"], (int, float)):
            raise ValueError(f"'{param_name}.max' should be an integer or float.")
        if not isinstance(range_params["step"], (int, float)):
            raise ValueError(f"'{param_name}.step' should be an integer or float.")
        
        if range_params["min"] >= range_params["max"]:
            raise ValueError(f"'{param_name}.min' should be less than '{param_name}.max'.")
        
        if range_params["step"] <= 0:
            raise ValueError(f"'{param_name}.step' should be a positive number.")
        
        start = range_params["min"]
        stop = range_params["max"]
        step = range_params["step"]
        
        if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
            return list(range(start, stop + 1, step))
        
        elif isinstance(start, (int, float)) and isinstance(stop, (int, float)) and isinstance(step, (int, float)):
            values = []
            current = start
            while current <= stop:
                values.append(round(current, 10))  
                current += step
            return values
        
        raise ValueError(f"Cannot generate range for '{param_name}'.")
    
    @staticmethod
    def validate_and_convert_color_dict(color_dict):
        if not isinstance(color_dict, dict):
            raise ValueError("'color_dict' must be a dictionary where keys are color lists as strings and values are integer labels.")

        converted_color_dict = {}

        for color_key, label in color_dict.items():
            try:
                color_tuple = tuple(map(int, color_key.strip("[]").split(",")))

                if len(color_tuple) != 3 or not all(0 <= c <= 255 for c in color_tuple):
                    raise ValueError(f"Invalid color '{color_key}'. Must be a list of three integers in the range [0, 255].")

                if not isinstance(label, int) or label <= 0:
                    raise ValueError(f"Invalid label '{label}' for color '{color_key}'. Must be a positive integer.")

                converted_color_dict[color_tuple] = label

            except ValueError:
                raise ValueError(f"Invalid color format '{color_key}'. Colors must be in the format '[R,G,B]' with integer values.")

        return converted_color_dict
    
    @staticmethod
    def load_config_file(file_path:str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        
        if not os.path.isfile(file_path):
            raise ValueError(f"File {file_path} is not valid.")
        
        file_extension = file_path.lower().split('.')[-1]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_extension == "json":
                    data = json.load(f)
                elif file_extension == "yaml" or file_extension == "yml":
                    data = yaml.safe_load(f)
                else:
                    raise ValueError(f"File {file_path} is neither a valid JSON nor YAML file.")
                
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"File {file_path} does not have a valid format. Error: {str(e)}")

        return ConfigHandler.read_config(data)

    @staticmethod
    def read_config(data: dict):
        
        for key in ConfigHandler.REQUIRED_KEYS:
            if key not in data:
                raise ValueError(f"Missing required configuration key: '{key}'.")
        
        if not os.path.exists(data["dataset_path"]):
            raise FileNotFoundError(f"Directory {data['dataset_path']} does not exist.")
        
        if not os.path.isdir(data["dataset_path"]):
            raise ValueError(f"Path {data['dataset_path']} is not a valid directory.")

        if not os.path.isdir(os.path.join(data["dataset_path"], "images")):
            raise ValueError(f"The dataset must include a subdirectory 'images' for the image files, and optionally a 'labels' subdirectory for the corresponding labels.")
        
        if not os.path.exists(data["output_path"]):
            raise FileNotFoundError(f"Output directory {data['dataset_path']} does not exist.")
        
        if not os.path.isdir(data["output_path"]):
            raise ValueError(f"Path {data['dataset_path']} is not a valid directory.")
        
        if not isinstance(data["analyze"], bool):
            raise ValueError(f"The value of 'analyze' must be an boolean (true or false), but got {type(data['analyze'])}.")
        
        if data.get("cluster_images"):
            if not all(fw in ConfigHandler.CONFIGURATION_VALUES["embedding_frameworks"] for fw in data.get("embedding_frameworks", [])):
                raise ValueError(f"Invalid embedding frameworks. Must be one of: {ConfigHandler.CONFIGURATION_VALUES['embedding_frameworks']}.")
            
            if "huggingface" in data["embedding_frameworks"] and "models_huggingface" not in data:
                raise ValueError(f"When 'huggingface' is selected in 'embedding_frameworks', a list of models must be specified under 'models_huggingface'.")
            
            if "pytorch" in data["embedding_frameworks"] and "models_pytorch" not in data:
                raise ValueError(f"When 'pytorch' is selected in 'embedding_frameworks', a list of models must be specified under 'models_pytorch'.")
            
            if "tensorflow" in data["embedding_frameworks"] and "models_tensorflow" not in data:
                raise ValueError(f"When 'tensorflow' is selected in 'embedding_frameworks', a list of models must be specified under 'models_tensorflow'.")
            
            for framework in ConfigHandler.CONFIGURATION_VALUES["embedding_frameworks"]:
                if framework not in data["embedding_frameworks"]:
                    model_key = f"models_{framework}"
                    if model_key in data:
                        raise ValueError(f"Models for '{framework}' should not be specified unless '{framework}' is included in 'embedding_frameworks'.")
            
            if any(framework in data["embedding_frameworks"] for framework in ["tensorflow", "opencv"]):
                if "resize_height" not in data:
                    print(f"Resize height not detected for {', '.join([framework for framework in ['tensorflow', 'opencv'] if framework in data['embedding_frameworks']])} model(s). Using default resize height of {ConfigHandler.DEFAULT_VALUES['resize_height']}.")
                    data['resize_height'] = ConfigHandler.DEFAULT_VALUES['resize_height']
                elif not isinstance(data["resize_height"], int):
                    raise ValueError(f"The value of 'resize_height' must be an integer, but got {type(data['resize_height'])}.")

                if "resize_width" not in data:
                    print(f"Resize width not detected for {', '.join([framework for framework in ['tensorflow', 'opencv'] if framework in data['embedding_frameworks']])} model(s). Using default resize width of {ConfigHandler.DEFAULT_VALUES['resize_width']}.")
                    data['resize_width'] = ConfigHandler.DEFAULT_VALUES['resize_width']
                elif not isinstance(data["resize_width"], int):
                    raise ValueError(f"The value of 'resize_width' must be an integer, but got {type(data['resize_width'])}.")
                
            if "opencv" in data["embedding_frameworks"]:
                if "lbp_radius" not in data:
                    print(f"Radius not detected for OpenCV LBP embedding model. Using default radius of {ConfigHandler.DEFAULT_VALUES['lbp_radius']}.")
                    data['lbp_radius'] = ConfigHandler.DEFAULT_VALUES['lbp_radius']
                elif not isinstance(data['lbp_radius'], int):
                    raise ValueError(f"The value of 'lbp_radius' must be an integer, but got {type(data['lbp_radius'])}.")
                
                if "lbp_num_points" not in data:
                    print(f"Number of points not detected for OpenCV LBP embedding model. Using default number of points of {ConfigHandler.DEFAULT_VALUES['lbp_num_points']}.")
                    data['lbp_num_points'] = ConfigHandler.DEFAULT_VALUES['lbp_num_points']
                elif not isinstance(data['lbp_num_points'], int):
                    raise ValueError(f"The value of 'lbp_num_points' must be an integer, but got {type(data['lbp_num_points'])}.")

                if "lbp_method" not in data:
                    print(f"LBP method not detected for OpenCV LBP embedding model. Using default method of {ConfigHandler.DEFAULT_VALUES['lbp_method']}.")
                    data['lbp_method'] = ConfigHandler.DEFAULT_VALUES['lbp_method']
                elif data.get("lbp_method") not in ConfigHandler.CONFIGURATION_VALUES['lbp_methods']:
                    raise ValueError(f"The value of 'lbp_method' must be one of {ConfigHandler.CONFIGURATION_VALUES['lbp_methods']}, but got {data['lbp_method']}.")

            if "clustering_models" not in data:
                raise ValueError(f"Clustering models configuration is missing.")
            
            for model, params in data['clustering_models'].items():
                
                if model not in ConfigHandler.CONFIGURATION_VALUES['clustering_models']:
                    raise ValueError(f"Invalid model '{model}' in 'clustering_models'. Must be one of: {ConfigHandler.CONFIGURATION_VALUES['clustering_models']}.")

                if model == "kmeans":
                    if "n_clusters_range" in params and "n_clusters" in params:
                        raise ValueError(f"Model '{model}' cannot have both 'n_clusters_range' and 'n_clusters' defined.")
                    
                    if "random_state" not in params:
                        print(f"Random_state not detected for KMeans clustering model. Using default random_state of {ConfigHandler.DEFAULT_VALUES['random_state']}")
                        data["clustering_models"][model]["random_state"] = ConfigHandler.DEFAULT_VALUES['random_state']

                    if "n_clusters_range" in params:
                        data["clustering_models"][model]["n_clusters_range"] = ConfigHandler.validate_range(params["n_clusters_range"], "n_clusters_range")

                elif model == "agglomerative":

                    if "n_clusters_range" in params and "n_clusters" in params:
                        raise ValueError(f"Model '{model}' cannot have both 'n_clusters_range' and 'n_clusters' defined.")
                    
                    if not "linkage" in params:
                        raise ValueError(f"You need to provide 'linkage' for the {model} model.")
                                        
                    if params.get("linkage") not in ConfigHandler.CONFIGURATION_VALUES["linkages"]:
                        raise ValueError(f"Invalid linkage '{params['linkage']}'. Must be one (or more) of {ConfigHandler.CONFIGURATION_VALUES['linkages']}.")
                    
                    if "n_clusters_range" in params:
                        data["clustering_models"][model]["n_clusters_range"] = ConfigHandler.validate_range(params["n_clusters_range"], "n_clusters_range")

                elif model == "dbscan":
                    if ("eps" in params or "min_samples" in params) and ("eps_range" in params or "min_samples_range" in params):
                        raise ValueError(f"Model '{model}' cannot mix fixed values and ranges for parameters 'eps' and 'min_samples'.")
                    
                    if "eps_range" in params:
                        data["clustering_models"][model]["eps_range"] = ConfigHandler.validate_range(params["eps_range"], "eps_range")
                    
                    if "min_samples_range" in params:
                        data["clustering_models"][model]["min_samples_range"] = ConfigHandler.validate_range(params["min_samples_range"], "min_samples_range")

                elif model == "optics":
                    if "min_samples" in params and "min_samples_range" in params:
                        raise ValueError(f"Model '{model}' cannot have both 'min_samples' and 'min_samples_range' defined.")
                    
                    if "min_samples_range" in params:
                        data["clustering_models"][model]["min_samples_range"] = ConfigHandler.validate_range(params["min_samples_range"], "min_samples_range")

            if "clustering_metric" not in data:
                print(f"clustering_metric not detected for clustering. Using default clustering_metric of {ConfigHandler.DEFAULT_VALUES['clustering_metric']}.")
                data["clustering_metric"] = ConfigHandler.DEFAULT_VALUES['clustering_metric']

            selected_models = list(data["clustering_models"].keys())

            clustering_metric = data["clustering_metric"]
            valid_metrics = ConfigHandler.CONFIGURATION_VALUES["clustering_metric"]  

            if "kmeans" in selected_models and len(selected_models) == 1:
                valid_metrics = ConfigHandler.CONFIGURATION_VALUES["kmeans_clustering_metric"]  

            if clustering_metric not in valid_metrics:
                raise ValueError(f"Invalid clustering_metric '{clustering_metric}'. Must be one of: {valid_metrics}.")
            
            if "plot" not in data:
                print(f"Plot not detected for clustering. Using default clustering_metric of {ConfigHandler.DEFAULT_VALUES['plot']}.")
                data["plot"] = ConfigHandler.DEFAULT_VALUES['plot']

            if not isinstance(data["plot"], bool):
                raise ValueError(f"The value of 'plot' must be an boolean (true or false), but got {type(data['plot'])}.")
            
            if data["plot"] and data.get("visualization_technique") not in ConfigHandler.CONFIGURATION_VALUES['visualization_techniques']:
                print(f"Visualization technique not detected when plot is set to True. Using default technique of {ConfigHandler.DEFAULT_VALUES['visualization_technique']}")
                data["visualization_technique"] = ConfigHandler.DEFAULT_VALUES['visualization_technique']
        
        if data.get("reduction"):

            if "reduction_percentage" not in data:
                raise ValueError(f"When 'reduction' is selected, a reduction percentage must be specified under 'reduction_percentage'.")
            
            if not isinstance(data["reduction_percentage"], float): 
                raise ValueError(f"The value of 'reduction_percentage' must be a float, but got {type(data['reduction_percentage'])}.")
            
            if data.get("reduction_type") not in ConfigHandler.CONFIGURATION_VALUES["reduction_type"]:
                raise ValueError(f"Invalid reduction type. Must be one of: {ConfigHandler.CONFIGURATION_VALUES['reduction_type']}.")
            
            if "diverse_percentage" not in data:
                print(f"Diverse percentage not detected for reduction. Using default diverse percentage of {ConfigHandler.DEFAULT_VALUES['diverse_percentage']}")
                data["diverse_percentage"] = ConfigHandler.DEFAULT_VALUES['diverse_percentage']

            if not isinstance(data["diverse_percentage"], float): 
                raise ValueError(f"The value of 'diverse_percentage' must be a float, but got {type(data['diverse_percentage'])}.")
        
            if data.get("reduction_model"):
                if isinstance(data["reduction_model"], str):
                    if data["reduction_model"] != "best_model":
                        raise ValueError(f"Invalid reduction model '{data['reduction_model']}'. Must be 'best_model'.")

                    if not data.get("cluster_images", False):
                        raise ValueError("'best_model' can only be used when 'cluster_images' is set to True.")
                
                elif isinstance(data["reduction_model"], dict):
                    model_keys = list(data["reduction_model"].keys())

                    if len(model_keys) != 1:
                        raise ValueError(f"The 'reduction_model' dictionary must contain exactly one model from {ConfigHandler.CONFIGURATION_VALUES['clustering_models']}.")

                    selected_model = model_keys[0]

                    if selected_model not in ConfigHandler.CONFIGURATION_VALUES["clustering_models"]:
                        raise ValueError(f"Invalid reduction model '{selected_model}'. Must be one of: {ConfigHandler.CONFIGURATION_VALUES['clustering_models']}.")
                    
                    model_params = data["reduction_model"][selected_model]

                    if not isinstance(data["reduction_model"][selected_model], dict):
                        raise ValueError(f"The parameters for '{selected_model}' must be provided as a dictionary.")
                    
                    if selected_model == "kmeans":
                        if "n_clusters" not in model_params or not isinstance(model_params["n_clusters"], int) or model_params["n_clusters"] <= 0:
                            raise ValueError("KMeans requires a positive integer 'n_clusters' parameter.")
                        if "random_state" in model_params and not isinstance(model_params["random_state"], int):
                            print(f"The 'random_state' parameter for KMeans must be an integer. Using default random_state of {ConfigHandler.DEFAULT_VALUES['random_state']}")
                            data["clustering_models"][model]["random_state"] = ConfigHandler.DEFAULT_VALUES['random_state']

                    elif selected_model == "agglomerative":
                        if "n_clusters" not in model_params or not isinstance(model_params["n_clusters"], int) or model_params["n_clusters"] <= 0:
                            raise ValueError("Agglomerative clustering requires a positive integer 'n_clusters' parameter.")
                        if "linkage" not in model_params or model_params["linkage"] not in ConfigHandler.CONFIGURATION_VALUES["linkages"]:
                            raise ValueError(f"Invalid 'linkage' parameter for Agglomerative clustering. Must be one of: {ConfigHandler.CONFIGURATION_VALUES['linkages']}.")

                    elif selected_model == "dbscan":
                        if "eps" not in model_params or not isinstance(model_params["eps"], (int, float)) or model_params["eps"] <= 0:
                            raise ValueError("DBSCAN requires a positive 'eps' parameter.")
                        if "min_samples" not in model_params or not isinstance(model_params["min_samples"], int) or model_params["min_samples"] <= 0:
                            raise ValueError("DBSCAN requires a positive integer 'min_samples' parameter.")

                    elif selected_model == "optics":
                        if "min_samples" not in model_params or not isinstance(model_params["min_samples"], int) or model_params["min_samples"] <= 0:
                            raise ValueError("OPTICS requires a positive integer 'min_samples' parameter.")
                            
                else:
                    raise ValueError("The 'reduction_model' must be either a string ('best_model') if cluster_images is True or a dictionary with one clustering model and its parameters if cluster_iamges is False.")
        
        if "depth_model" not in data:
            print(f"Depth model for YOLO/COCO label transformation not defined. The default model {ConfigHandler.DEFAULT_VALUES['depth_model']} from HuggingFace will be used if needed.")
            data["depth_model"] = ConfigHandler.DEFAULT_VALUES['depth_model']

        if not isinstance(data["depth_model"], str):
            raise ValueError(f"The value of 'depth_model' must be a string, but got {type(data['depth_model'])}.")
        
        if "threshold" not in data:
            print(f"Threshold for binary label transformation not defined. The default threshold {ConfigHandler.DEFAULT_VALUES['threshold']} will be used if needed.")
            data["threshold"] = ConfigHandler.DEFAULT_VALUES['threshold']

        if not isinstance(data["threshold"], int):
            raise ValueError(f"The value of 'threshold' must be an integer, but got {type(data['threshold'])}.")
        
        if "color_dict" not in data:
            print(f"Color dictionary for multicolor label transformation not defined. It will be calculated automatically if needed.")
        else: 
            data["color_dict"] = ConfigHandler.validate_and_convert_color_dict(data["color_dict"])

        if not isinstance(data["split_percentages"], dict):
            raise ValueError("'split_percentages' must be a dictionary with keys 'train', 'valid', and 'test'.")

        required_keys = {"train", "valid", "test"}
        if set(data["split_percentages"].keys()) != required_keys:
            raise ValueError(f"'split_percentages' must contain exactly the keys: {required_keys}.")

        if not all(isinstance(data["split_percentages"][key], float) for key in required_keys):
            raise ValueError("'split_percentages' values must all be floats.")

        if not all(0 <= data["split_percentages"][key] <= 1 for key in required_keys):
            raise ValueError("Each 'split_percentages' value must be between 0 and 1.")

        if round(sum(data["split_percentages"].values()), 10) != 1.0:
            raise ValueError("The sum of 'train', 'valid', and 'test' split percentages must be exactly 1.0.")

        if data.get("stratification") and data.get("stratification_type") not in ConfigHandler.CONFIGURATION_VALUES['stratification_types']:
            print(f"Invalid stratification_type selected. Must be one of: {ConfigHandler.CONFIGURATION_VALUES['stratification_types']}. Selected default value of {ConfigHandler.DEFAULT_VALUES['stratification_type']}")
            data["stratification_type"] = ConfigHandler.DEFAULT_VALUES['stratification_type']

        if "cross_validation" in data:
            if not isinstance(data["cross_validation"], int):
                raise ValueError("'cross_validation' must be an integer.")
            if data["cross_validation"] < 0:
                raise ValueError("'cross_validation' must be greater than or equal to 0.")
            if data["cross_validation"] > 0 and ("train" not in data["split_percentages"] or "valid" not in data["split_percentages"] or "test" not in data["split_percentages"]):
                raise ValueError("When cross-validation is enabled,'train', 'valid' and 'test' splits must be provided in 'split_percentages'.")    

        if data.get("segmentation") not in ConfigHandler.CONFIGURATION_VALUES["segmentation"]:
            raise ValueError(f"Invalid segmentation type. Must be one of: {ConfigHandler.CONFIGURATION_VALUES['segmentation']}.")
        
        if data["segmentation"] == "semantic" and data.get('models') not in ConfigHandler.CONFIGURATION_VALUES["semantic_segmentation_models"]: 
            raise ValueError(f"Invalid semantic segmentation model. Must be one of: {ConfigHandler.CONFIGURATION_VALUES['semantic_segmentation_models']}.")
        
        if data["segmentation"] == "instance" and data.get('models') not in ConfigHandler.CONFIGURATION_VALUES["instance_segmentation_models"]:
            raise ValueError(f"Invalid instance segmentation model. Must be of: {ConfigHandler.CONFIGURATION_VALUES['instance_segmentation_models']}.")
        
        if data.get("segmentation_metric") not in ConfigHandler.CONFIGURATION_VALUES["segmentation_metric"]:
            raise ValueError(f"Invalid evaluation metric. Must be one of {ConfigHandler.CONFIGURATION_VALUES['segmentation_metric']}.")
        
        if not isinstance(data["batch_size"], int): 
            raise ValueError(f"The value of 'batch_size' must be an integer, but got {type(data['batch_size'])}.")
        
        if not isinstance(data["epochs"], int): 
            raise ValueError(f"The value of 'epochs' must be an integer, but got {type(data['epochs'])}.")
        
        if not isinstance(data["background"], int): 
            raise ValueError(f"The value of 'background' must be an integer, but got {type(data['background'])}.")
        
        if not isinstance(data["verbose"], bool):
            raise ValueError(f"The value of 'verbose' must be a boolean (true or false), but got {type(data['verbose'])}.")
        
        if data["verbose"]:
            print("Successfully loaded data from config file.")

        return data
