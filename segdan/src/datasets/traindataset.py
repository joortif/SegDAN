import numpy as np
import pandas as pd
import os
import shutil

from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import train_test_split

from PIL import Image

from sklearn.utils import shuffle

import cv2

class TrainDataset():

    def __init__(self, img_path, label_path, output_path):
        self.img_path = img_path
        self.label_path = label_path
        self.output_path = output_path

        self.image_files = sorted([os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.jpg')])
        self.mask_paths = []

    def calculate_class_distributions(self, mask_paths, num_classes: int, background: int | None = None):
        class_distributions = []
        for mask_path in mask_paths:
            with Image.open(mask_path) as mask_image:
                mask = np.array(mask_image)
                unique, counts = np.unique(mask, return_counts=True)

                if background is not None:
                    mask_exclude_idx = np.where(unique == background)[0]
                    if len(mask_exclude_idx) > 0: 
                        unique = np.delete(unique, mask_exclude_idx)
                        counts = np.delete(counts, mask_exclude_idx)

                adjusted_indices = [
                    class_id if background is None or class_id < background else class_id - 1
                    for class_id in unique
                ]

                distribution = np.zeros(num_classes)
                for adjusted_class_id, count in zip(adjusted_indices, counts):
                    if adjusted_class_id < num_classes:  
                        distribution[adjusted_class_id] = count

                class_distributions.append(distribution / distribution.sum())
        
        return np.array(class_distributions)

    def calculate_object_number(self, masks, num_classes: int, background: int | None = None):
        all_objects_per_class = []

        for mask_path in masks:
            with Image.open(mask_path) as mask_image:
                mask = np.array(mask_image)
                
                unique_classes = np.unique(mask)
                
                objects_per_class = np.zeros(num_classes)

                for class_id in unique_classes:
                    
                    if background is not None and class_id == background:
                        continue

                    adjusted_class_id = class_id if background is None or class_id < background else class_id - 1

                    if adjusted_class_id >= num_classes:
                        continue

                    class_mask = np.where(mask == class_id, 255, 0).astype(np.uint8)

                    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    objects_per_class[adjusted_class_id] = len(contours)
                all_objects_per_class.append(objects_per_class)

        return np.array(all_objects_per_class)
    
    def calculate_pixel_ratio(self, masks: str, num_classes: int, background: int = None) -> np.ndarray:
        objects_per_class = self.calculate_object_number(masks, num_classes, background)

        all_pixel_ratios = []

        for i, mask_path in enumerate(masks):
            with Image.open(mask_path) as mask_image:
                mask = np.array(mask_image)

                pixels_per_class = np.zeros(num_classes)

                unique_classes = np.unique(mask)

                for class_id in unique_classes:
                    if background is not None and class_id == background:
                        continue

                    adjusted_class_id = class_id if background is None or class_id < background else class_id - 1

                    if adjusted_class_id >= num_classes:
                        continue

                    pixels_per_class[adjusted_class_id] = np.sum(mask == class_id)

                pixel_ratios = np.zeros(num_classes)
                for class_id in range(num_classes):
                    if objects_per_class[i, class_id] > 0:
                        pixel_ratios[class_id] = pixels_per_class[class_id] / objects_per_class[i, class_id]

                all_pixel_ratios.append(pixel_ratios)

        return np.array(all_pixel_ratios)
    
    def save_split_to_directory(self, train_df, val_df, test_df):
        subsets = {"train": train_df, "val": val_df, "test": test_df}

        for subset, df in subsets.items():
            if df is None:
                continue
            img_dir = os.path.join(self.output_path, subset, "images")
            label_dir = os.path.join(self.output_path, subset, "labels")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            for _, row in df.iterrows():
                shutil.copy(row["img_path"], os.path.join(img_dir, os.path.basename(row["img_path"])))
                shutil.copy(row["mask_path"], os.path.join(label_dir, os.path.basename(row["mask_path"])))

    def split(self, num_classes: int, train_fraction: float, val_fraction: float, test_fraction: float, stratify: True,
        stratification_strategy: str = "pixel_distribution",  background: int | None = None, random_state: int=123
    ):
        
        if stratify and stratification_strategy.lower() not in ["pixel_distribution", "num_objects", "pixel_objects_ratio"]:
            raise ValueError("Invalid value for stratification_strategy. Must be 'pixel_distribution', 'num_objects' or 'pixel_objects_ratio'.")
        
        if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
            raise ValueError("The sum of train_fraction, val_fraction, and test_fraction must equal 1.0")

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("The number of images and masks must be the same.")
        
        if stratify:
            train_images, train_masks, val_images, val_masks, test_images, test_masks = self.stratify_split(self.image_paths, self.mask_paths, num_classes, train_fraction, val_fraction, test_fraction, stratification_strategy, background, random_state)
        else:
            train_images, train_masks, val_images, val_masks, test_images, test_masks = self.random_split(self.image_paths, self.mask_paths, train_fraction, val_fraction, test_fraction, random_state)

        
        train_df = pd.DataFrame({"img_path": train_images, "mask_path": train_masks})
        val_df = pd.DataFrame({"img_path": val_images, "mask_path": val_masks})
        test_df = pd.DataFrame({"img_path": test_images, "mask_path": test_masks})

        self.save_split_to_directory(train_df, val_df, test_df)

        
    def stratify_split(self, image_paths, mask_paths, num_classes, train_fraction, val_fraction, test_fraction, stratification_strategy, background, random_state):
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
        image_paths, mask_paths = shuffle(image_paths, mask_paths, random_state=random_state)

        distributions = []

        if stratification_strategy.lower() == "pixel_distribution":
            distributions = self.calculate_class_distributions(mask_paths, num_classes, background)
        elif stratification_strategy.lower() == "num_objects":
            distributions = self.calculate_object_number(mask_paths, num_classes, background)
        else:
            distributions = self.calculate_pixel_ratio(mask_paths, num_classes, background)

        # Split the data into train and "everything else" using IterativeStratification
        stratifier = IterativeStratification(
            n_splits=2,
            order=1,
            sample_distribution_per_fold=[train_fraction, 1.0 - train_fraction]
        )

        everything_else_indexes, train_indexes = next(stratifier.split(X=np.zeros(len(image_paths)), y=distributions))

        # Separate the image and mask paths into train and everything_else sets
        train_images = [image_paths[i] for i in train_indexes]
        train_masks = [mask_paths[i] for i in train_indexes]

        everything_else_images = [image_paths[i] for i in everything_else_indexes]
        everything_else_masks = [mask_paths[i] for i in everything_else_indexes]

        # Step 2: Split the "everything else" set into validation and test
        val_proportion = val_fraction / (val_fraction + test_fraction)  

        stratifier = IterativeStratification(
            n_splits=2,
            order=1,
            sample_distribution_per_fold=[val_proportion, 1.0 - val_proportion]
        )

        test_indexes, val_indexes  = next(stratifier.split(X=np.zeros(len(everything_else_images)), y=distributions[everything_else_indexes]))

        # Separate the image and mask paths into val and test sets
        val_images = [everything_else_images[i] for i in val_indexes]
        val_masks = [everything_else_masks[i] for i in val_indexes]
        test_images = [everything_else_images[i] for i in test_indexes]
        test_masks = [everything_else_masks[i] for i in test_indexes]

        return train_images, train_masks, val_images, val_masks, test_images, test_masks
    
    def random_split(self, image_paths, mask_paths, train_fraction, val_fraction, test_fraction, random_state):
        
        train_val_images, test_images, train_val_masks, test_masks = train_test_split(
            image_paths, mask_paths, test_size=test_fraction, random_state=random_state
        )

        val_fraction_adjusted = val_fraction / (train_fraction + val_fraction)  
        train_images, val_images, train_masks, val_masks = train_test_split(
            train_val_images, train_val_masks, test_size=val_fraction_adjusted, random_state=random_state
        )

        return train_images, train_masks, val_images, val_masks, test_images, test_masks
    
    
    def automate_split_nested(
        self,
        image_paths: str,
        mask_paths: str,
        num_classes: int,
        test_fraction: float,
        n_splits: int,
        fold_val_fraction: float,
        output_dir: str | None = None,
        random_state: int = 42
        ):
        """
        Automatiza el proceso de partición en un conjunto de test fijo + K-Fold en train/val.

        Args:
            image_paths: Lista de rutas de las imágenes.
            mask_paths: Lista de rutas de las máscaras.
            num_classes: Número de clases en el dataset (para estratificación).
            output_dir: Directorio donde se guardarán las particiones. Si None, no se guardan.
            test_fraction: Fracción del dataset para el conjunto de test.
            n_splits: Número de folds para K-Fold split en train/val.
            stratified: Si True, usa estratificación para los splits.
            random_state: Semilla para la aleatorización.

        Returns:
            - test_df: DataFrame del conjunto de test.
            - folds: Lista de tuplas [(train_df, val_df)] para cada fold.
        """
        if len(image_paths) != len(mask_paths):
            raise ValueError("The number of images and masks must be the same.")

        image_paths, mask_paths = shuffle(image_paths, mask_paths, random_state=random_state)

        class_distributions = self.calculate_class_distributions(mask_paths, num_classes)

        # Split the data into train and "everything else" using IterativeStratification
        stratifier = IterativeStratification(
            n_splits=2,
            order=1,
            sample_distribution_per_fold=[test_fraction, 1.0 - test_fraction]
        )

        everything_else_indexes, test_indexes = next(stratifier.split(X=np.zeros(len(image_paths)), y=class_distributions))

        test_images = [image_paths[i] for i in test_indexes]
        test_masks = [mask_paths[i] for i in test_indexes]

        train_val_images = [image_paths[i] for i in everything_else_indexes]
        train_val_masks = [mask_paths[i] for i in everything_else_indexes]

        test_df = pd.DataFrame({"img_path": test_images, "mask_path": test_masks})
        print(np.full(n_splits, fold_val_fraction))
        # K-Fold en el resto (train + val)
        folds = []
        stratifier = IterativeStratification(
            n_splits=n_splits,
            order=1,
            sample_distribution_per_fold=[fold_val_fraction] * n_splits
        )

        used_val_indices = set()

        # Realizamos el KFold con la distribución de clases para el resto de los datos
        for fold_idx, (train_idx, val_idx) in enumerate(stratifier.split(X=np.zeros(len(train_val_images)), y=class_distributions[everything_else_indexes])):

            if any(idx in used_val_indices for idx in val_idx):
                raise ValueError(f"Índices duplicados encontrados en el fold {fold_idx + 1} para el conjunto de validación.")
            
            used_val_indices.update(val_idx)

            print(f"Fold {fold_idx + 1}:")
            print(f"  Training indices: {train_idx}...")  # Muestra los primeros 5 índices de entrenamiento
            print(f"  Validation indices: {val_idx}...\n")  # Muestra los primeros 5 índices de validación
            
            train_images = [train_val_images[i] for i in train_idx]
            train_masks = [train_val_masks[i] for i in train_idx]
            val_images = [train_val_images[i] for i in val_idx]
            val_masks = [train_val_masks[i] for i in val_idx]

            train_df = pd.DataFrame({"img_path": train_images, "mask_path": train_masks})
            val_df = pd.DataFrame({"img_path": val_images, "mask_path": val_masks})
            folds.append((train_df, val_df))

            if output_dir:
                fold_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
                self.save_split_to_directory(train_df, val_df, None, fold_dir)

        if output_dir:
            test_dir = os.path.join(output_dir, "test")
            os.makedirs(test_dir, exist_ok=True)
            self.save_split_to_directory(None, None, test_df, output_dir)

        return test_df, folds