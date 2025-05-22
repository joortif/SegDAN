import os 
import numpy as np

from src.utils.constants import SegmentationType
from src.models.smpmodel import SMPModel
from src.models.hfsemanticmodel import HFFormerModel

from src.datasets.hfdataset import HuggingFaceAdapterDataset
from src.datasets.smpdataset import SMPDataset
from torch.utils.data import DataLoader


from src.utils.confighandler import ConfigHandler
from src.datasets.augments import get_training_augmentation, get_validation_augmentation


smp_models_lower = [name.lower() for name in ConfigHandler.SEMANTIC_SEGMENTATION_MODELS["smp"]]
hf_models_lower = [name.lower() for name in ConfigHandler.SEMANTIC_SEGMENTATION_MODELS["hf"]]

def model_training(model_data: dict, general_data:dict, split_path: str, model_output_path: str, label_path: str, hold_out: bool, classes: int | None):

    epochs = model_data["epochs"]
    batch_size = model_data["batch_size"]
    evaluation_metrics = model_data["evaluation_metrics"]
    selection_metric = model_data["selection_metric"]
    segmentation_type = model_data["segmentation"]
    models = model_data["models"]

    background = general_data["background"]
    

    os.makedirs(model_output_path, exist_ok=True)
    
    if segmentation_type == SegmentationType.SEMANTIC.value:
        models = rename_model_sizes(models)
        semantic_model_training(epochs, batch_size, evaluation_metrics, selection_metric, models, split_path, hold_out, classes, background, model_output_path)

    
    return

def rename_model_sizes(models: np.ndarray):
    
    for model in models:
        model_size = model["model_size"]
        model_name = model["model_name"]

        if model_name in smp_models_lower:
            model["model_size"] = ConfigHandler.CONFIGURATION_VALUES["model_sizes_smp"].get(model_size)

        if model_name in hf_models_lower:
            model["model_size"] = ConfigHandler.CONFIGURATION_VALUES["model_sizes_hf"].get(model_size)

    return models


def get_data_splits(split_path: str, hold_out: bool, classes: int):
    
    if hold_out:
        train_path = os.path.join(split_path, "train") 
        val_path = os.path.join(split_path, "val") if os.path.exists(os.path.join(split_path, "val")) else None
        test_path = os.path.join(split_path, "test")

        yield {
            "train": SMPDataset(os.path.join(train_path, "images"), os.path.join(train_path, "labels"), classes, get_training_augmentation),
            "val": SMPDataset(os.path.join(val_path, "images"), os.path.join(val_path, "labels"), classes, get_training_augmentation) if val_path else None,
            "test": SMPDataset(os.path.join(test_path, "images"), os.path.join(test_path, "labels"), classes, get_validation_augmentation)
        }
    else:
        fold_dirs = sorted([f for f in os.listdir(split_path) if f.startswith("fold_")])
        for fold in fold_dirs:
            fold_path = os.path.join(split_path, fold)
            train_path = os.path.join(fold_path, "train")
            val_path = os.path.join(fold_path, "val")

            yield {
                "train": SMPDataset(os.path.join(train_path, "images"), os.path.join(train_path, "labels"), classes, get_training_augmentation),
                "val": SMPDataset(os.path.join(val_path, "images"), os.path.join(val_path, "labels"), classes, get_validation_augmentation),
                "test": None
            }

def semantic_model_training(epochs: int, batch_size:int, evaluation_metrics: np.ndarray, selection_metric:str, models: np.ndarray, split_path: str, hold_out: bool, classes: int, background: int | None, output_path: str):

    for fold_idx, data_split in enumerate(get_data_splits(split_path, hold_out, classes)):
        train_dataset = data_split["train"]
        val_dataset = data_split["val"]
        test_dataset = data_split["test"]

        for model_config in models:
            model_size = model_config["model_size"]
            model_name = model_config["model_name"]

            if model_name in hf_models_lower:
                model = HFFormerModel(model_name, model_size, classes, evaluation_metrics, selection_metric, epochs, batch_size)

                train_adapter = HuggingFaceAdapterDataset(train_dataset, model.feature_extractor)
                val_adapter = HuggingFaceAdapterDataset(val_dataset, model.feature_extractor) if val_dataset else None
                test_adapter = HuggingFaceAdapterDataset(test_dataset, model.feature_extractor)

                model.train(train_adapter, val_adapter, test_adapter)

            else:
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True) if val_dataset else None
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True) if test_dataset else None

                T_MAX = epochs * len(train_loader)
                out_classes = len([cls for cls in classes if cls.lower() !="background"])

                model = SMPModel(3, out_classes, evaluation_metrics, T_MAX, background, model_name, model_size)

                model.train(epochs, train_loader, val_loader, test_loader, os.path.join(output_path, "metrics.csv"))

            

    
