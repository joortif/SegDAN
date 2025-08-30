import os
from typing import Optional 
import numpy as np
from torch.utils.data import DataLoader

from segdan.utils.constants import SegmentationType
from segdan.models.smpmodel import SMPModel
from segdan.models.hfstransformermodel import HFTransformerModel

from segdan.datasets.hfdataset import HuggingFaceAdapterDataset
from segdan.datasets.smpdataset import SMPDataset


from segdan.utils.confighandler import ConfigHandler
from segdan.utils.utils import Utils
from segdan.datasets.augments import get_training_augmentation, get_validation_augmentation


smp_models_lower = [name.lower() for name in ConfigHandler.SEMANTIC_SEGMENTATION_MODELS["smp"]]
hf_models_lower = [name.lower() for name in ConfigHandler.SEMANTIC_SEGMENTATION_MODELS["hf"]]

def model_training(model_data: dict, general_data:dict, split_path: str, model_output_path: str, hold_out: bool, classes: Optional[np.ndarray], mode_height: int, mode_width:int):

    epochs = model_data["epochs"]
    evaluation_metrics = model_data["evaluation_metrics"]
    selection_metric = model_data["selection_metric"]
    segmentation_type = model_data["segmentation"]
    models = model_data["models"]
    background = general_data["background"]

    os.makedirs(model_output_path, exist_ok=True)
    
    if segmentation_type == SegmentationType.SEMANTIC.value:
        models = rename_model_sizes(models)

        resize_shape = Utils.calculate_closest_resize(mode_height, mode_width)

        semantic_model_training(epochs=epochs, imgsz=resize_shape, evaluation_metrics=evaluation_metrics, selection_metric=selection_metric, models=models, 
                                split_path=split_path, hold_out=hold_out, classes=classes, background=background, output_path=model_output_path)

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

def get_data_splits(split_path: str, hold_out: bool, classes: np.ndarray, imgsz:int, background: Optional[int]):
    
    if hold_out:
        train_path = os.path.join(split_path, "train") 
        val_path = os.path.join(split_path, "val") if os.path.exists(os.path.join(split_path, "val")) else None
        test_path = os.path.join(split_path, "test")

        yield {
            "train": SMPDataset(os.path.join(train_path, "images"), os.path.join(train_path, "labels"), classes, get_training_augmentation(imgsz, imgsz), background),
            "val": SMPDataset(os.path.join(val_path, "images"), os.path.join(val_path, "labels"), classes, get_training_augmentation(imgsz, imgsz), background) if val_path else None,
            "test": SMPDataset(os.path.join(test_path, "images"), os.path.join(test_path, "labels"), classes, get_validation_augmentation(imgsz, imgsz), background)
        }
    else:
        fold_dirs = sorted([f for f in os.listdir(split_path) if f.startswith("fold_")])
        for fold in fold_dirs:
            fold_path = os.path.join(split_path, fold)
            train_path = os.path.join(fold_path, "train")
            val_path = os.path.join(fold_path, "val")

            yield {
                "train": SMPDataset(os.path.join(train_path, "images"), os.path.join(train_path, "labels"), classes, get_training_augmentation(), background),
                "val": SMPDataset(os.path.join(val_path, "images"), os.path.join(val_path, "labels"), classes, get_validation_augmentation(), background),
                "test": None
            }

def semantic_model_training(epochs: int, imgsz:int, evaluation_metrics: np.ndarray, selection_metric:str, models: np.ndarray, split_path: str, hold_out: bool, classes: np.ndarray, background: Optional[int], output_path: str):

    best_metric = -float("inf")
    best_model_path = None
    best_model_name = None

    for fold_idx, data_split in enumerate(get_data_splits(split_path=split_path, hold_out=hold_out, classes=classes, imgsz=imgsz, background=background)):
        train_dataset = data_split["train"]
        val_dataset = data_split["val"]
        test_dataset = data_split["test"]

        for model_config in models:
            model_size = model_config["model_size"]
            model_name = model_config["model_name"]

            if model_name in hf_models_lower:
                model = HFTransformerModel(model_name=model_name, model_size=model_size, classes=classes, metrics=evaluation_metrics,
                                           selection_metric=selection_metric, epochs=epochs, imgsz=imgsz, output_path=output_path)

                train_loader = HuggingFaceAdapterDataset(train_dataset, model.feature_extractor)
                val_loader = HuggingFaceAdapterDataset(val_dataset, model.feature_extractor) if val_dataset else None
                test_loader = HuggingFaceAdapterDataset(test_dataset, model.feature_extractor)
            else:

                model = SMPModel(in_channels=3, classes=classes, metrics=evaluation_metrics, imgsz=imgsz, selection_metric=selection_metric, epochs=epochs, t_max=None, output_path=output_path, model_name=model_name, encoder_name=model_size)

                train_loader = DataLoader(train_dataset, batch_size=model.batch, shuffle=True, num_workers=4, persistent_workers=True)
                val_loader = DataLoader(val_dataset, batch_size=model.batch, shuffle=False, num_workers=4, persistent_workers=True) if val_dataset else None
                test_loader = DataLoader(test_dataset, batch_size=model.batch, shuffle=False, num_workers=4, persistent_workers=True)

                model.t_max = epochs * len(train_loader)
                
            print(f"Training {model_name} - {model_size}...")
                
            evaluation_metric, candidate_path = model.run_training(train_loader, val_loader, test_loader)

            if evaluation_metric > best_metric:
                print(f"New best model found: {model_name} with {selection_metric} score of {evaluation_metric}")

                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)

                best_model_name = model_name
                best_metric = evaluation_metric
                best_model_path = candidate_path
            else:
                if os.path.exists(candidate_path):
                    os.remove(candidate_path)
                    
            print(f"Best model: {best_model_name}")
            print(f"{selection_metric.capitalize()} score: {best_metric}")

    return best_model_path  
