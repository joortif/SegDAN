from src.datasets.traindataset import TrainingDataset
import os

def dataset_split(dataset: TrainingDataset, general_data: dict, split_data: dict):

    split_method = split_data["split_method"]
    
    hold_out = split_data.get("hold_out", {})
    train_percentage = hold_out.get("train", None)
    valid_percentage = hold_out.get("valid", None)
    test_percentage = hold_out.get("test", None)

    cross_val = split_data.get("cross_val", {})
    n_folds = cross_val.get("num_folds", None)

    stratify = split_data.get("stratification", False)
    random_seed = split_data.get("stratification_random_seed", 123)

    background = general_data.get("background", None)

    stratification_type = split_data.get("stratification_type", None)
    
    dataset.split(general_data, split_data)
    return
