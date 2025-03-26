import numpy as np
import pandas as pd
import os
import shutil
import json
from tqdm import tqdm

from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import KFold, train_test_split

from PIL import Image

from sklearn.utils import shuffle

import cv2

class TrainingDataset():

    def __init__(self, img_path, multilabel_label_path, output_path, label_format, original_label_path):
        self.img_path = img_path
        self.label_path = multilabel_label_path
        self.output_path = output_path
        self.label_format = label_format
        self.original_label_path = original_label_path

        self.image_files = sorted([os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.jpg')])
        
        self.mask_paths_multilabel = sorted([os.path.join(multilabel_label_path, f) for f in os.listdir(multilabel_label_path) if f.endswith('.png')])


    def calculate_pixel_distribution(self, mask_paths, background: int | None = None):
        all_distributions = []
        unique_classes = set()

        for mask_path in tqdm(mask_paths, desc="Calculating pixel distribution for each class"):
            with Image.open(mask_path) as mask_image:
                mask = np.array(mask_image)
                unique, counts = np.unique(mask, return_counts=True)

                if background is not None:
                    mask_exclude_idx = np.where(unique == background)[0]
                    if len(mask_exclude_idx) > 0: 
                        unique = np.delete(unique, mask_exclude_idx)
                        counts = np.delete(counts, mask_exclude_idx)

                unique_classes.update(unique)

                distribution = {class_id: count for class_id, count in zip(unique, counts)}
                all_distributions.append(distribution)
                
        num_classes = len(unique_classes)
        sorted_classes = sorted(unique_classes)

        matrix_distributions = np.zeros((len(mask_paths), num_classes))

        for i, distribution in enumerate(all_distributions):
            for class_id, count in distribution.items():
                class_index = sorted_classes.index(class_id)
                matrix_distributions[i, class_index] = count

        row_sums = matrix_distributions.sum(axis=1, keepdims=True)  
        matrix_distributions = np.divide(matrix_distributions, row_sums, where=row_sums != 0)  
        
        return {
            "distributions": matrix_distributions,
            "num_classes": num_classes
        }

    def calculate_object_number(self, masks, background: int | None = None):
        all_objects_per_class = []
        unique_classes = set()

        for mask_path in tqdm(masks, desc="Reading number of classes from images"):
            with Image.open(mask_path) as mask_image:
                mask = np.array(mask_image)
                unique = np.unique(mask)

                if background is not None:
                    unique = unique[unique != background]

                unique_classes.update(unique)

        sorted_classes = sorted(unique_classes)  
        num_classes = len(sorted_classes)
        class_mapping = {class_id: idx for idx, class_id in enumerate(sorted_classes)}

        for mask_path in tqdm(masks, desc="Calculating number of objects for each class"):
            with Image.open(mask_path) as mask_image:
                mask = np.array(mask_image)

                objects_per_class = np.zeros(num_classes) 

                for class_id in np.unique(mask):
                    if background is not None and class_id == background:
                        continue

                    if class_id in class_mapping:
                        class_mask = np.where(mask == class_id, 255, 0).astype(np.uint8)
                        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        objects_per_class[class_mapping[class_id]] = len(contours)

                all_objects_per_class.append(objects_per_class)

        return {
            "distributions": np.array(all_objects_per_class),
            "num_classes": num_classes,
            "class_mapping": class_mapping
        }
    
    def calculate_pixel_ratio(self, masks, background: int | None = None):
        object_info = self.calculate_object_number(masks, background)
        objects_per_class = object_info["distributions"]  
        num_classes = object_info["num_classes"]
        class_mapping = object_info["class_mapping"]

        all_pixel_ratios = []

        for i, mask_path in tqdm(enumerate(masks), total=len(masks), desc="Calculating pixel-to-object ratio for each class"):
            with Image.open(mask_path) as mask_image:
                mask = np.array(mask_image)

                pixels_per_class = np.zeros(num_classes)

                for class_id, class_index in class_mapping.items():
                    if background is not None and class_id == background:
                        continue  

                    pixels_per_class[class_index] = np.sum(mask == class_id)

                pixel_ratios = np.zeros(num_classes)
                for class_id, class_index in class_mapping.items():
                    if background is not None and class_id == background:
                        continue  

                    if objects_per_class[i, class_index] > 0:
                        pixel_ratios[class_index] = pixels_per_class[class_index] / objects_per_class[i, class_index]

                all_pixel_ratios.append(pixel_ratios)

        return {
            "distributions": np.array(all_pixel_ratios),
            "num_classes": num_classes,
        }
    
    def save_mask_format(self, row, img_dir, label_dir):
        shutil.copy(row["img_path"], os.path.join(img_dir, os.path.basename(row["img_path"])))
        shutil.copy(row["mask_path"], os.path.join(label_dir, os.path.basename(row["mask_path"])))
    
    def save_txt_format(self, row, img_dir, label_dir):
        shutil.copy(row["img_path"], os.path.join(img_dir, os.path.basename(row["img_path"])))
        txt_label_path = row["mask_path"].replace(".png", ".txt").replace(".jpg", ".txt")
        txt_label_path = os.path.join(self.original_label_path, os.path.basename(txt_label_path))
        if os.path.exists(txt_label_path):
            shutil.copy(txt_label_path, os.path.join(label_dir, os.path.basename(txt_label_path)))

    def save_json_format(self, row, img_dir, label_dir, coco_data, selected_images, selected_annotations, image_ids):
        img_name = os.path.basename(row["img_path"])
        for img in coco_data["images"]:
            if img["file_name"] == img_name:
                selected_images.append(img)
                current_img_id = img["id"]
                image_ids.add(current_img_id)
                break
        for ann in coco_data["annotations"]:
            if ann["image_id"] == current_img_id:
                selected_annotations.append(ann)
        shutil.copy(row["img_path"], os.path.join(img_dir, os.path.basename(row["img_path"])))        

    def save_split_to_directory(self, train_df, val_df, test_df, fold_dir: str = None):

        print("Saving splits...")

        subsets = {"train": train_df, "val": val_df, "test": test_df}

        format_handlers = {
            "mask": self.save_mask_format,
            "txt": self.save_txt_format,
            "json": None  
        }

        output_path = self.output_path

        if fold_dir:
            output_path = fold_dir

        for subset, df in subsets.items():
            if df is None:
                continue
            
            img_dir = os.path.join(output_path, subset, "images")
            label_dir = os.path.join(output_path, subset, "labels")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            if self.label_format in format_handlers:
                handler = format_handlers[self.label_format]
                if handler:
                    tqdm.pandas(desc=f"Saving {subset} images and labels")
                    df.progress_apply(lambda row: handler(row, img_dir, label_dir), axis=1)
                    continue  
                    
            with open(self.original_label_path, "r") as f:
                coco_data = json.load(f)

            selected_images = []
            selected_annotations = []
            image_ids = set()
            
            tqdm.pandas(desc=f"Saving {subset} JSON annotations")
            df.progress_apply(lambda row: self.save_json_format(row, img_dir, label_dir, coco_data, selected_images, selected_annotations, image_ids), axis=1)
        
            coco_subset = {
                "images": selected_images,
                "annotations": selected_annotations,
                "categories": coco_data["categories"]
            }
            with open(os.path.join(label_dir, f"{subset}.json"), "w") as f:
                json.dump(coco_subset, f, separators=(',', ':'))

        print(f"Splits successfully saved in {output_path}")
        
    def split(self, hold_out: bool, train_fraction: float = 0.7, val_fraction: float = 0.2, test_fraction: float = 0.1, n_folds: int=5, 
              stratify: bool=True, stratification_strategy: str = "pixels",  background: int | None = None, random_state: int=123
    ):
        
        if stratify and stratification_strategy.lower() not in ["pixels", "objects", "pixel_to_object_ratio"]:
            raise ValueError("Invalid value for stratification_strategy. Must be 'pixels', 'objects' or 'pixel_to_object_ratio'.")
        
        if hold_out:
            if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
                raise ValueError("The sum of train_fraction, val_fraction, and test_fraction must equal 1.0")

        if self.mask_paths_multilabel != None and len(self.image_files) != len(self.mask_paths_multilabel):
            raise ValueError("The number of images and masks must be the same.")
        
        if hold_out: 
            if stratify:
                train_images, train_masks, val_images, val_masks, test_images, test_masks = self.stratify_split(train_fraction, val_fraction, test_fraction, stratification_strategy, background, random_state)
            else:
                train_images, train_masks, val_images, val_masks, test_images, test_masks = self.random_split(train_fraction, val_fraction, test_fraction, random_state)
        
            train_df = pd.DataFrame({"img_path": train_images, "mask_path": train_masks})
            val_df = pd.DataFrame({"img_path": val_images, "mask_path": val_masks})
            test_df = pd.DataFrame({"img_path": test_images, "mask_path": test_masks})

            self.save_split_to_directory(train_df, val_df, test_df)

            return
        
        self.kfold_cross_validation(n_folds, stratify, stratification_strategy, background, random_state)

        
    def stratify_split(self, train_fraction, val_fraction, test_fraction, stratification_strategy, background, random_state):
        """
        Stratify-shuffle-split a semantic segmentation dataset into
        train/val/test sets based on pixel-wise class distributions and save
        the results in specified directories.

        Args:
            image_paths: List of file paths to the input images.
            mask_paths: List of file paths to the corresponding segmentation masks.
            train_fraction: Fraction of data to reserve for the training dataset.
            val_fraction: Fraction of data to reserve for the validation dataset.
            test_fraction: Fraction of data to reserve for the test dataset.
            output_dir: Parent directory where the train, val, and test directories will be created. If None, the directories will not be created. Defaults to None.

        Returns:
            Tuple containing three DataFrames: train, val, and test subsets.
            Each DataFrame has two columns: 'img_path' and 'mask_path'.
        """
        image_paths, mask_paths = shuffle(self.image_files, self.mask_paths_multilabel, random_state=random_state)

        print("Starting classes stratification...")
        if stratification_strategy.lower() == "pixels":
            result = self.calculate_pixel_distribution(mask_paths, background)
        elif stratification_strategy.lower() == "objects":
            result = self.calculate_object_number(mask_paths, background)
        else:
            result = self.calculate_pixel_ratio(mask_paths, background)
        
        distributions = result["distributions"]
        num_classes = result["num_classes"]

        print(f"Stratification done. {num_classes} classes detected.")
        
        stratifier = IterativeStratification(
            n_splits=2,
            order=1,
            sample_distribution_per_fold=[train_fraction, 1.0 - train_fraction]
        )

        everything_else_indexes, train_indexes = next(stratifier.split(X=np.zeros(len(image_paths)), y=distributions))

        train_images = [image_paths[i] for i in train_indexes]
        train_masks = [mask_paths[i] for i in train_indexes]

        everything_else_images = [image_paths[i] for i in everything_else_indexes]
        everything_else_masks = [mask_paths[i] for i in everything_else_indexes]

        val_proportion = val_fraction / (val_fraction + test_fraction)  

        stratifier = IterativeStratification(
            n_splits=2,
            order=1,
            sample_distribution_per_fold=[val_proportion, 1.0 - val_proportion]
        )

        test_indexes, val_indexes  = next(stratifier.split(X=np.zeros(len(everything_else_images)), y=distributions[everything_else_indexes]))

        val_images = [everything_else_images[i] for i in val_indexes]
        val_masks = [everything_else_masks[i] for i in val_indexes]
        test_images = [everything_else_images[i] for i in test_indexes]
        test_masks = [everything_else_masks[i] for i in test_indexes]

        return train_images, train_masks, val_images, val_masks, test_images, test_masks
    
    def random_split(self, train_fraction, val_fraction, test_fraction, random_state):
        
        train_val_images, test_images, train_val_masks, test_masks = train_test_split(
            self.image_files, self.mask_paths_multilabel, test_size=test_fraction, random_state=random_state
        )

        val_fraction_adjusted = val_fraction / (train_fraction + val_fraction)  
        train_images, val_images, train_masks, val_masks = train_test_split(
            train_val_images, train_val_masks, test_size=val_fraction_adjusted, random_state=random_state
        )

        return train_images, train_masks, val_images, val_masks, test_images, test_masks
    
    
    def kfold_cross_validation(
        self,
        n_splits: int,
        stratify: bool = False, 
        stratification_strategy: str = "pixels",
        background: int = None,
        random_state: int = 42
        ):
        """
        Automates the partitioning process into a fixed test set + K-Fold for train/val.

        Args:
            output_dir: Directory where the partitions will be saved. If None, they are not saved.
            test_fraction: Fraction of the dataset allocated to the test set.
            n_splits: Number of folds for K-Fold split in train/val.
            random_state: Seed for randomization.

        Returns:
            - test_df: DataFrame containing the test set.
            - folds: List of tuples [(train_df, val_df)] for each fold.
        """
        if len(self.image_files) != len(self.mask_paths_multilabel):
            raise ValueError("The number of images and masks must be the same.")

        image_paths, mask_paths = shuffle(self.image_files, self.mask_paths_multilabel, random_state=random_state)

        used_val_indices = set()
        folds = []

        if not stratify:
            print("Saving splits...")

            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

            for fold_idx, (train_idx, val_idx) in tqdm(enumerate(kfold.split(X=np.zeros(len(image_paths)), y=np.zeros(len(image_paths)))), 
                                                       total=n_splits, 
                                                       desc="Processing folds"):

                if any(idx in used_val_indices for idx in val_idx):
                    raise ValueError(f"Duplicate indices found in fold {fold_idx + 1} for the validation set.")
                
                used_val_indices.update(val_idx)
                
                train_images = [image_paths[i] for i in train_idx]
                train_masks = [mask_paths[i] for i in train_idx]
                val_images = [image_paths[i] for i in val_idx]
                val_masks = [mask_paths[i] for i in val_idx]

                train_df = pd.DataFrame({"img_path": train_images, "mask_path": train_masks})
                val_df = pd.DataFrame({"img_path": val_images, "mask_path": val_masks})
                folds.append((train_df, val_df))

                fold_dir = os.path.join(self.output_path, f"fold_{fold_idx + 1}")
                self.save_split_to_directory(train_df, val_df, None, fold_dir)
                
            return

        print("Starting stratification...")

        if stratification_strategy.lower() == "pixels":
            result = self.calculate_pixel_distribution(mask_paths, background)
        elif stratification_strategy.lower() == "objects":
            result = self.calculate_object_number(mask_paths, background)
        else:
            result = self.calculate_pixel_ratio(mask_paths, background)

        distributions = result["distributions"]
        num_classes = result["num_classes"]

        print(f"Stratification done. {num_classes} classes detected.")


        stratifier = IterativeStratification(
            n_splits=n_splits,
            order=1,
            sample_distribution_per_fold=[1.0 / n_splits] * n_splits
        )

        for fold_idx, (train_idx, val_idx) in tqdm(enumerate(stratifier.split(X=np.zeros(len(image_paths)), 
                                                                      y=distributions)), 
                                           total=n_splits, 
                                           desc="Processing folds"):

            if any(idx in used_val_indices for idx in val_idx):
                raise ValueError(f"Duplicate indices found in fold {fold_idx + 1} for the validation set.")
            
            used_val_indices.update(val_idx)
            
            train_images = [image_paths[i] for i in train_idx]
            train_masks = [mask_paths[i] for i in train_idx]
            val_images = [image_paths[i] for i in val_idx]
            val_masks = [mask_paths[i] for i in val_idx]

            train_df = pd.DataFrame({"img_path": train_images, "mask_path": train_masks})
            val_df = pd.DataFrame({"img_path": val_images, "mask_path": val_masks})
            folds.append((train_df, val_df))

            fold_dir = os.path.join(self.output_path, f"fold_{fold_idx + 1}")
            self.save_split_to_directory(train_df, val_df, None, fold_dir)

        return
