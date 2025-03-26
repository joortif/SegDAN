from src.datasets.traindataset import TrainingDataset
import os

def dataset_split(dataset: TrainingDataset, general_data: dict, training_data: dict):

    split_method = training_data["split_method"]
    
    hold_out = training_data.get("hold_out", {})
    train_percentage = hold_out.get("train", None)
    valid_percentage = hold_out.get("valid", None)
    test_percentage = hold_out.get("test", None)

    cross_val = training_data.get("cross_val", {})
    n_folds = cross_val.get("num_folds", None)

    stratify = training_data.get("stratification", False)
    random_seed = training_data.get("stratification_random_seed", 123)

    background = general_data.get("background", None)

    stratification_type = training_data.get("stratification_type", None)
    
    dataset.split(split_method, train_percentage, valid_percentage, test_percentage, n_folds, stratify, stratification_type, background, random_seed)
    return
