import os
import logging
import pandas as pd
import shutil
from typing import Optional 
import numpy as np
from torch.utils.data import DataLoader

from segdan.utils.constants import SegmentationType
from segdan.models.smpmodel import SMPModel
from segdan.models.hfstransformermodel import HFTransformerModel
from segdan.models.yolomodel import YOLOModel

from segdan.datasets.hfdataset import HFDataset
from segdan.datasets.smpdataset import SMPDataset

from segdan.utils.confighandler import ConfigHandler
from segdan.utils.utils import Utils
from segdan.datasets.augments import get_training_augmentation, get_validation_augmentation



logger = logging.getLogger(__name__)

smp_models_lower = [name.lower() for name in ConfigHandler.SEMANTIC_SEGMENTATION_MODELS["smp"]]
hf_models_lower = [name.lower() for name in ConfigHandler.SEMANTIC_SEGMENTATION_MODELS["hf"]]

def model_training(model_data: dict, general_data:dict, split_path: str, model_output_path: str, hold_out: bool, classes: Optional[np.ndarray], mode_height: int, mode_width:int):

    epochs = int(model_data["epochs"])
    evaluation_metrics = model_data["evaluation_metrics"]

    selection_metric = (
        model_data.get("segmentation_selection_metric") or model_data.get("selection_metric") or "map"
    )

    if hasattr(selection_metric, "get"):
        selection_metric = selection_metric.get()

    segmentation_type = model_data["segmentation"]
    models = model_data["models"]
    background = general_data["background"]

    os.makedirs(model_output_path, exist_ok=True)

    imgsz = Utils.calculate_closest_resize(mode_height, mode_width)
    
    if segmentation_type == SegmentationType.SEMANTIC.value:
        models = rename_model_sizes(models)

        return semantic_model_training(epochs=epochs, imgsz=imgsz, evaluation_metrics=evaluation_metrics, selection_metric=selection_metric, models=models, 
                                split_path=split_path, hold_out=hold_out, classes=classes, background=background, output_path=model_output_path)

    if segmentation_type == SegmentationType.INSTANCE.value:

        batch_size = model_data.get("batch_size", general_data.get("batch_size", 8))

        if hasattr(batch_size, "get"):
            batch_size = batch_size.get()

        return instance_model_training(epochs=epochs, imgsz=imgsz, evaluation_metrics=evaluation_metrics, selection_metric=selection_metric, models=models,
                                split_path=split_path, hold_out=hold_out, classes=classes, background=background, output_path=model_output_path, batch_size=batch_size)
    raise ValueError(f"Unsupported segmentation type: {segmentation_type}. Supported types are: {SegmentationType.SEMANTIC.value}, {SegmentationType.INSTANCE.value}")



def rename_model_sizes(models: np.ndarray):
    
    for model in models:
        model_size = model["model_size"]
        model_name = model["model_name"]

        if model_name in smp_models_lower:
            model["model_size"] = ConfigHandler.CONFIGURATION_VALUES["model_sizes_smp"].get(model_size)

        if model_name in hf_models_lower:
            model["model_size"] = ConfigHandler.CONFIGURATION_VALUES["model_sizes_hf"].get(model_size)

    return models

def get_augment(imgsz, split: str="train"):
    if split.lower() == "train":
        return get_training_augmentation(imgsz, imgsz)

    return get_validation_augmentation(imgsz, imgsz)

def build_model(model_name: str, model_size: str, model_type: str, classes, evaluation_metrics, selection_metric, epochs: int, imgsz: int, output_path: str):
    if model_type=="hf":
        model = HFTransformerModel(model_name=model_name, model_size=model_size, classes=classes, metrics=evaluation_metrics,
                                           selection_metric=selection_metric, epochs=epochs, imgsz=imgsz, output_path=output_path)
    elif model_type=="smp":
        model = SMPModel(in_channels=3, classes=classes, metrics=evaluation_metrics, imgsz=imgsz, selection_metric=selection_metric,
                                            epochs=epochs, t_max=None, output_path=output_path, model_name=model_name, encoder_name=model_size)
        
    return model

def build_dataset(model_type:str, split_path: str, batch_size:int, classes, imgsz, background):

    train_path = os.path.join(split_path, "train") if os.path.exists(os.path.join(split_path, "train")) else None
    val_path = os.path.join(split_path, "val") if os.path.exists(os.path.join(split_path, "val")) else None
    test_path = os.path.join(split_path, "test") if os.path.exists(os.path.join(split_path, "test")) else None

    binary = True if len(classes) == 2 else False

    if model_type == 'hf':
        train_ds = HFDataset(os.path.join(train_path, "images"), os.path.join(train_path, "labels"), classes, 
                             augmentation=get_training_augmentation(imgsz, imgsz), background=background, binary=binary) if train_path else None
        val_ds = HFDataset(os.path.join(val_path, "images"), os.path.join(val_path, "labels"), classes, 
                             augmentation=get_validation_augmentation(imgsz, imgsz), background=background, binary=binary) if val_path else None
        test_ds = HFDataset(os.path.join(test_path, "images"), os.path.join(test_path, "labels"), classes, 
                             augmentation=get_validation_augmentation(imgsz, imgsz), background=background, binary=binary) if test_path else None

        return train_ds, val_ds, test_ds

    if model_type == 'smp':
        train_ds = SMPDataset(os.path.join(train_path, "images"), os.path.join(train_path, "labels"), classes, 
                             augmentation=get_training_augmentation(imgsz, imgsz), background=background, binary=binary) if train_path else None
        val_ds = SMPDataset(os.path.join(val_path, "images"), os.path.join(val_path, "labels"), classes, 
                             augmentation=get_validation_augmentation(imgsz, imgsz), background=background, binary=binary) if val_path else None
        test_ds = SMPDataset(os.path.join(test_path, "images"), os.path.join(test_path, "labels"), classes, 
                             augmentation=get_validation_augmentation(imgsz, imgsz), background=background, binary=binary) if test_path else None


        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4) if train_path else None
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4) if val_ds else None
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4) if test_path else None
        return train_loader, val_loader, test_loader

    # TODO: YOLO or other semantic models
    raise ValueError(f"Unsupported model type {model_type}")

def semantic_model_training(epochs: int, imgsz:int, evaluation_metrics: np.ndarray, selection_metric:str, models: np.ndarray, split_path: str, hold_out: bool, classes: np.ndarray, background: Optional[int], output_path: str):

    best_metric = -float("inf")
    best_model_path = None
    best_model_name = None

    for model_config in models:
        model_name = model_config["model_name"]
        model_size = model_config["model_size"]

        if model_name in hf_models_lower:
            model_type="hf"
        elif model_name in smp_models_lower:
            model_type="smp"
        else:
            raise ValueError(f"Model {model_name} is not among HuggingFace Transformers models: {hf_models_lower} or SMP models: {smp_models_lower}")

        if hold_out:

            model = build_model(model_name=model_name, model_size=model_size, model_type=model_type, classes=classes, 
                            evaluation_metrics=evaluation_metrics, selection_metric=selection_metric, epochs=epochs, imgsz=imgsz, output_path=output_path)

            model.autobatch_imgsz()
            batch_size = model.batch

            train, val, test = build_dataset(model_type=model_type, split_path=split_path, batch_size=batch_size, classes=classes,imgsz=imgsz, background=background)

            if model_type=="smp":
                model.t_max = epochs * len(train)
            
            model.init_trainer(train, val, test)

            logger.info(f"Training {model_name} - {model_size}... -> Hold out")

            evaluation_metric, candidate_path = model.run_training()

            if evaluation_metric > best_metric:
                logger.info(f"New best model found: {model_name} with {selection_metric} score of {evaluation_metric}")

                Utils.safe_remove(best_model_path)

                best_model_name = model_name
                best_metric = evaluation_metric
                best_model_path = candidate_path
            else:
                Utils.safe_remove(candidate_path)
                        
        else:
            fold_names = sorted([f for f in os.listdir(split_path) if f.startswith("fold_")])
            fold_dirs = [os.path.join(split_path, f) for f in fold_names]

            for fold_idx, fold in enumerate(fold_dirs): 
                
                model = build_model(model_name=model_name, model_size=model_size, model_type=model_type, classes=classes, 
                            evaluation_metrics=evaluation_metrics, selection_metric=selection_metric, epochs=epochs, imgsz=imgsz, output_path=output_path)

                model.autobatch_imgsz()
                batch_size = model.batch

                train, val, _ = build_dataset(model_type=model_type, split_path=fold, batch_size=batch_size, classes=classes, imgsz=imgsz, background=background)
                _, _, test = build_dataset(model_type=model_type, split_path=split_path, batch_size=batch_size, classes=classes, imgsz=imgsz, background=background)
            
                if model_type=="smp":
                    model.t_max = epochs * len(train)
                
                model.init_trainer(train, val, test)

                logger.info(f"Training {model_name} - {model_size}... -> Fold {fold_idx} of {len(fold_dirs)}")

                evaluation_metric, candidate_path = model.run_training()

                if evaluation_metric > best_metric:
                    logger.info(f"New best model found: {model_name} with {selection_metric} score of {evaluation_metric}")

                    Utils.safe_remove(best_model_path)

                    best_model_name = model_name
                    best_metric = evaluation_metric
                    best_model_path = candidate_path
                else:
                    Utils.safe_remove(candidate_path)
        
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"{selection_metric.capitalize()} score: {best_metric}")
        
    return best_model_path

def instance_model_training(epochs: int, imgsz:int, evaluation_metrics: np.ndarray, selection_metric:str, models: np.ndarray, split_path: str, hold_out: bool, classes: np.ndarray, background: Optional[int], output_path: str, batch_size:int):
    if not hold_out:
        raise NotImplementedError("Cross-validation is not implemented for instance segmentation models.")

    if isinstance(models, str):
        models = [{"model_name": models, "model_size": None}]

    os.makedirs(output_path, exist_ok=True)

    best_metric = -float("inf")
    best_model_path = None
    best_model_name = None
    validation_rows = []

    for model_config in models:
        model_name = str(model_config["model_name"]).lower()
        model_size = str(model_config.get("model_size", "small")).lower()

        if model_name not in {"yolo", "yolo11", "yolo11-seg"}:
            raise ValueError(f"Model {model_name} not implemented yet")

        model = YOLOModel(classes=classes, epochs=epochs, imgsz=imgsz, metrics=evaluation_metrics, selection_metric=selection_metric,
                          output_path=output_path, model_name=model_name, model_size=model_size, batch_size=batch_size)

        run_name = f"yolo11-{model.model_size}-seg"

        score, candidate_path = model.run_training(split_path=split_path, run_name=run_name, test_root=split_path)

        validation_rows.append({
            "model": model.weights_name,
            "selection_metric": selection_metric,
            "validation_score": score,
            **{
                f"{key}_val": value
                for key, value in model.validation_metrics.items()
            },
            "training_time": model.training_time,
            "checkpoint_path": str(candidate_path)
        })

        if score > best_metric:
            best_metric = score
            best_model = model
            best_model_path = candidate_path

        if best_model is None:
            raise RuntimeError("No best model found. Training may have failed.")

        pd.DataFrame(validation_rows).to_csv(os.path.join(output_path, "yolo_validation_metrics.csv"), sep=";", decimal=",", index=False)


        logger.info(f"Best model: {best_model.weights_name}, validation {selection_metric} = {best_metric:.4f}")

        return best_model_path