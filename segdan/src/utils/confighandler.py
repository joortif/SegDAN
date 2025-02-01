import json
import os

class ConfigHandler():

    REQUIRED_KEYS = [
        "dataset_path", 
        "output_path", 
        "segmentation", 
        "analyze",
        "models", 
        "metric", 
        "batch_size", 
        "epochs",
        "verbose"
        ]


    CONFIGURATION_VALUES = {
        "segmentation": ["semantic", "instance"],
        "semantic_segmentation_models": ["unet", "deeplabv3"],
        "instance_segmentation_models": ["yolo"],
        "metric": ["iou", "dice_score"]
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
    def read_config(json_path: str):

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File {json_path} does not exist.")
        
        if not os.path.isfile(json_path):
            raise ValueError(f"File {json_path} is not valid.")

        try: 
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"File {json_path} does not have a valid JSON format.")
        
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
            raise FileNotFoundError(f"Directory {data['dataset_path']} does not exist.")
        
        if not os.path.isdir(data["output_path"]):
            raise ValueError(f"Path {data['dataset_path']} is not a valid directory.")
        
        if not isinstance(data["analyze"], bool):
            raise ValueError(f"The value of 'analyze' must be an boolean (true or false), but got {type(data['analyze'])}.")
        
        if data["segmentation"] not in ConfigHandler.CONFIGURATION_VALUES["segmentation"]:
            raise ValueError(f"Invalid segmentation type. Must be one of: {ConfigHandler.CONFIGURATION_VALUES['segmentation']}.")
        
        if data["segmentation"] == "semantic" and data["models"] not in ConfigHandler.CONFIGURATION_VALUES["semantic_segmentation_models"]: 
            raise ValueError(f"Invalid semantic segmentation model. Must be one of: {ConfigHandler.CONFIGURATION_VALUES['semantic_segmentation_models']}.")
        
        if data["segmentation"] == "instance" and data["models"] not in ConfigHandler.CONFIGURATION_VALUES["instance_segmentation_models"]:
            raise ValueError(f"Invalid instance segmentation model. Must be of: {ConfigHandler.CONFIGURATION_VALUES['instance_segmentation_models']}.")
        
        if data["metric"] not in ConfigHandler.CONFIGURATION_VALUES["metric"]:
            raise ValueError(f"Invalid evaluation metric. Must be one of {ConfigHandler.CONFIGURATION_VALUES['metric']}-")
        
        if not isinstance(data["batch_size"], int): 
            raise ValueError(f"The value of 'batch_size' must be an integer, but got {type(data['batch_size'])}.")
        
        if not isinstance(data["epochs"], int): 
            raise ValueError(f"The value of 'epochs' must be an integer, but got {type(data['epochs'])}.")
        
        if not isinstance(data["background"], int): 
            raise ValueError(f"The value of 'background' must be an integer, but got {type(data['background'])}.")
        
        if not isinstance(data["verbose"], bool):
            raise ValueError(f"The value of 'verbose' must be an boolean (true or false), but got {type(data['verbose'])}.")
        
        return data
