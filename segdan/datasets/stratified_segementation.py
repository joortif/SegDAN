from typing import List, Optional
import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm
from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import train_test_split

from PIL import Image

from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from imagedatasetanalyzer import ImageLabelDataset

import cv2

if __name__ == "__main__":

    
    def calculate_pixel_distribution(mask_paths, background: Optional[int] = None):
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

    def calculate_object_number( masks, background: Optional[int] = None):
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
    
    def calculate_pixel_ratio(masks, background: Optional[int] = None):
        object_info = calculate_object_number(masks, background)
        objects_per_class = object_info["distributions"]  
        num_classes = object_info["num_classes"]
        class_mapping = object_info["class_mapping"]

        all_pixel_ratios = []

        for i, mask_path in tqdm(enumerate(masks), total=len(masks), desc="Calculating pixel-to-object ratio for each class"):
            with Image.open(mask_path) as mask_image:
                mask = np.array(mask_image)
                total_pixels = mask.size

                pixels_per_class = np.zeros(num_classes)

                for class_id, class_index in class_mapping.items():
                    if background is not None and class_id == background:
                        continue  

                    pixels_per_class[class_index] = np.sum(mask == class_id)

                pixel_ratios = np.zeros(num_classes)
                for class_id, class_index in class_mapping.items():
                    if background is not None and class_id == background:
                        continue  

                    num_objects = objects_per_class[i, class_index]

                    if num_objects > 0:
                        pixel_ratios[class_index] = pixels_per_class[class_index] / (num_objects * total_pixels)
                    else:
                        pixel_ratios[class_index] = 0.0

                all_pixel_ratios.append(pixel_ratios)

        return {
            "distributions": np.array(all_pixel_ratios),
            "num_classes": num_classes,
        }
    
    
    def check_class_distribution(output_dir: str, dataset_name: str, num_classes: int):
        """
        Verifica y visualiza la distribución de clases en los directorios de train, val y test
        según el porcentaje de píxeles que ocupan en las máscaras.
        También guarda los resultados en un CSV y genera un gráfico de barras agrupadas.
        """

        splits = ['train', 'val', 'test']
        distributions = {}
        class_image_counts = {split: {cls: 0 for cls in range(num_classes)} for split in splits}
        total_class_counts = {cls: 0 for cls in range(num_classes)}

        all_files = set()
        for split in splits:
            label_dir = os.path.join(output_dir, split, 'labels')
            mask_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.png')]

            # Check for duplicate files between splits
            for mask_file in mask_files:
                if mask_file in all_files:
                    raise ValueError(f"Duplicate file found across splits: {mask_file}")
                all_files.add(mask_file)

            class_counts = np.zeros(num_classes)
            total_pixels = 0

            for mask_file in mask_files:
                mask = np.array(Image.open(mask_file))
                unique, counts = np.unique(mask, return_counts=True)
                class_counts[unique] += counts
                total_pixels += mask.size

                for cls in unique:
                    class_image_counts[split][cls] += 1
                    total_class_counts[cls] += 1

            class_distribution = class_counts / total_pixels
            distributions[split] = class_distribution

            print(f"Class distribution for {split}:")
            for cls_idx, dist in enumerate(class_distribution):
                print(f"  Class {cls_idx}: {dist:.4f}")
            print("\n")

        # Guardar a CSV
        distributions_clean = {split: dist.tolist() for split, dist in distributions.items()}

        df = pd.DataFrame.from_dict(distributions_clean, orient='index')
        df.columns = [f"Class_{i}" for i in range(num_classes)]
        df.index.name = 'Split'

        # Guardar CSV
        csv_path = os.path.join(output_dir, f"{dataset_name}_pixel_distribution.csv")
        df.to_csv(csv_path,sep=';', encoding="utf8")
        print(f"[CSV] Guardado a {csv_path}")
        
    def calculate_class_distributions(mask_paths: list[str], num_classes: int):
        class_distributions = []
        for mask_path in mask_paths:
            mask = np.array(Image.open(mask_path))
            unique, counts = np.unique(mask, return_counts=True)
            distribution = np.zeros(num_classes)
            distribution[unique] = counts
            class_distributions.append(distribution / distribution.sum())
        
        return np.array(class_distributions)

    def check_class_distribution_with_folds(output_dir: str, dataset_name: str, num_classes: int, n_splits: int):
        """
        Verifica la distribución de clases en los directorios de test y en los folds (train, val) para
        comprobar que la estratificación es correcta.

        Args:
            output_dir: Directorio padre donde se encuentran las carpetas test y folds.
            dataset_name: Nombre del dataset (usado para el nombre del archivo de salida).
            num_classes: Número total de clases en las máscaras.
            n_splits: Número total de folds.
        """

        splits = ['test'] + [f'fold_{i+1}' for i in range(n_splits)]  # Incluye 'test' y los folds
        distributions = {}
        class_image_counts = {split: {cls: 0 for cls in range(num_classes)} for split in splits + [f'fold_{i+1}_train' for i in range(n_splits)] + [f'fold_{i+1}_val' for i in range(n_splits)]}
        total_class_counts = {cls: 0 for cls in range(num_classes)}

        all_files = set()

        # Recorre los splits
        for split in splits:
            if split == 'test':
                # Directorio de test
                label_dir = os.path.join(output_dir, 'test', 'labels')
                mask_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.png')]
                distributions[split] = calculate_class_distributions(mask_files, num_classes)
            else:
                # Directorios de cada fold (train y val)
                label_dir_train = os.path.join(output_dir, split, 'train', 'labels')
                label_dir_val = os.path.join(output_dir, split, 'val', 'labels')
                mask_files_train = [os.path.join(label_dir_train, f) for f in os.listdir(label_dir_train) if f.endswith('.png')]
                mask_files_val = [os.path.join(label_dir_val, f) for f in os.listdir(label_dir_val) if f.endswith('.png')]

                # Crear subdistribuciones para train y val
                distributions[f'{split}_train'] = calculate_class_distributions(mask_files_train, num_classes)
                distributions[f'{split}_val'] = calculate_class_distributions(mask_files_val, num_classes)

        # Mostrar y graficar la distribución
        plot_class_distribution(distributions, num_classes, dataset_name, output_dir)

    def plot_class_distribution(distributions, class_names, num_classes, dataset_name, output_dir):
        bar_width = 0.35
        bar_spacing = 0.2  # Espaciado entre grupos de barras
        group_spacing = 0.6  # Espaciado entre clases

        num_splits = len(distributions)
        total_width = num_splits * bar_width + (num_splits - 1) * bar_spacing

        x_positions = []
        labels = []

        # Crear posiciones y etiquetas para las barras
        for i, split in enumerate(distributions.keys()):
            x_positions.append(np.arange(num_classes) * (total_width + group_spacing) + i * (bar_width + bar_spacing))
            labels.append(split)

        plt.figure(figsize=(16, 8))

        for i, (split, distribution) in enumerate(distributions.items()):
            offset = (i - (num_splits - 1) / 2) * bar_width
            for dist in distribution:
                plt.bar(x_positions[i], distribution, bar_width, label=split)
            

        # Etiquetas y formato
        plt.xlabel('Class')
        plt.ylabel('Pixel proportion')
        plt.title('Class distribution per fold (train/val) and test')
        plt.xticks(np.arange(num_classes) * (total_width + group_spacing) + total_width / 2 - bar_width / 2, class_names)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Guardar y mostrar la gráfica
        output = os.path.join(output_dir, f"{dataset_name}_class_distribution.png")
        plt.savefig(output, format='png')
        print(f"Plot saved to {output}")

        plt.show()
        plt.close()
    
    def save_split_to_directory(train_df, val_df, test_df, output_dir):
        """
        Saves the partitions into structured directories.
        """
        subsets = {"train": train_df, "val": val_df, "test": test_df}

        for subset, df in subsets.items():
            if df is None:
                continue
            img_dir = os.path.join(output_dir, subset, "images")
            label_dir = os.path.join(output_dir, subset, "labels")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            for _, row in df.iterrows():
                shutil.copy(row["img_path"], os.path.join(img_dir, os.path.basename(row["img_path"])))
                shutil.copy(row["mask_path"], os.path.join(label_dir, os.path.basename(row["mask_path"])))

    def check_objects_per_class(output_dir: str, class_names, dataset_name: str, num_classes: int, background: int = None):
        """
        Verifica el número de objetos por clase en las imágenes de los directorios train, val y test
        y genera un diagrama de barras combinando todos los splits en un solo gráfico con colores consistentes para cada clase.

        Args:
            output_dir (str): Directorio donde se encuentran las carpetas train, val y test.
            num_classes (int): Número total de clases en las máscaras.
            background (Optional[int], opcional): Clase de fondo a excluir. Por defecto es None.
        """
        splits = ['train', 'val', 'test']
        objects_per_class_by_split = {}


        # Iterar sobre los splits y calcular los objetos por clase
        for i, split in enumerate(splits):
            label_dir = os.path.join(output_dir, split, 'labels')
            mask_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.png')]

            # Obtener el número de objetos por clase usando calculate_object_number
            objects_per_class = calculate_object_number(mask_files, background)
            
            # Sumar el total de objetos por clase para todas las imágenes en el split
            total_objects = objects_per_class["distributions"].sum(axis=0)

            for j, k in enumerate(total_objects):
                print(f"Total objects from class {j} in split {split}: {k}" )

            # Guardar los resultados por split
            objects_per_class_by_split[split] = total_objects

        df = pd.DataFrame.from_dict(objects_per_class_by_split, orient='index').astype(int)
        df.columns = [f"Class_{i}" for i in range(num_classes)]
        df.index.name = 'Split'

        # Guardar CSV
        csv_path = os.path.join(output_dir, f"{dataset_name}_objects_per_class.csv")
        df.to_csv(csv_path, sep=';')
        print(f"[CSV] Guardado a {csv_path}")

        # csv_path = os.path.join(output_dir, f"{dataset_name}_objects_per_class.csv")
        # df = pd.DataFrame(objects_per_class_by_split)
        # df.index = class_names
        # df.to_csv(csv_path, sep=';')
        # print(f"[CSV] Guardado a {csv_path}")

    def check_ratio_per_class(output_dir: str, class_names, dataset_name: str, background: int = None):
        splits = ['train', 'val', 'test']
        objects_per_class_by_split = {}

        # Iterar sobre los splits y calcular los objetos por clase
        for split in splits:
            label_dir = os.path.join(output_dir, split, 'labels')
            mask_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.png')]

            pixel_ratios = calculate_pixel_ratio(mask_files, background)
            avg_ratios = np.nanmean(pixel_ratios["distributions"], axis=0)
            objects_per_class_by_split[split] = avg_ratios

        # Crear DataFrame con valores absolutos
        df_raw = pd.DataFrame(objects_per_class_by_split)
        df_raw.index = class_names

        df_percent = df_raw * 100

        percent_csv_path = os.path.join(output_dir, f"{dataset_name}_pixel_to_object_ratios_percent.csv")
        df_percent.to_csv(percent_csv_path, sep=';', float_format="%.3f", decimal=',', encoding="utf-8")
        print(f"[CSV] Guardado porcentajes en: {percent_csv_path}")

    def stratify_split(image_files, mask_paths_multilabel,train_fraction, val_fraction, test_fraction, stratification_strategy, background, random_state):
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
        image_paths, mask_paths = shuffle(image_files, mask_paths_multilabel, random_state=random_state)

        print("Starting classes stratification...")
        if stratification_strategy.lower() == "pixels":
            result = calculate_pixel_distribution(mask_paths, background)
        elif stratification_strategy.lower() == "objects":
            result = calculate_object_number(mask_paths, background)
        else:
            result = calculate_pixel_ratio(mask_paths, background)
        
        distributions = result["distributions"]
        num_classes = result["num_classes"]

        print(f"Stratification done. {num_classes} classes detected.")
        
        stratifier = IterativeStratification(
            n_splits=2,
            order=1,
            sample_distribution_per_fold=[train_fraction, 1.0 - train_fraction]
        )

        everything_else_indexes, train_indexes = next(stratifier.split(X=np.zeros(len(image_paths)), y=distributions))
        print("Train:", len(train_indexes))
        print("Val+Test:", len(everything_else_indexes))
    
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
        print("Val:", len(val_indexes))
        print("Test:", len(test_indexes))

        val_images = [everything_else_images[i] for i in val_indexes]
        val_masks = [everything_else_masks[i] for i in val_indexes]
        test_images = [everything_else_images[i] for i in test_indexes]
        test_masks = [everything_else_masks[i] for i in test_indexes]

        return train_images, train_masks, val_images, val_masks, test_images, test_masks, num_classes
    
    def random_split(image_files, mask_paths_multilabel, train_fraction, val_fraction, test_fraction, random_state):
        
        train_val_images, test_images, train_val_masks, test_masks = train_test_split(
            image_files,mask_paths_multilabel, test_size=test_fraction, random_state=random_state
        )

        val_fraction_adjusted = val_fraction / (train_fraction + val_fraction)  
        train_images, val_images, train_masks, val_masks = train_test_split(
            train_val_images, train_val_masks, test_size=val_fraction_adjusted, random_state=random_state
        )

        return train_images, train_masks, val_images, val_masks, test_images, test_masks

    def split(image_files, mask_paths_multilabel, num_classes, train_fraction=0.7, val_fraction=0.2, test_fraction=0.1, stratify=True, hold_out=True, stratification_strategy="pixel_objects_ratio", background=None, output_dir=None, random_state=4):
        
        if stratify and stratification_strategy.lower() not in ["pixels", "objects", "pixel_to_object_ratio"]:
            raise ValueError("Invalid value for stratification_strategy. Must be 'pixels', 'objects' or 'pixel_to_object_ratio'.")
        
        if hold_out:
            if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
                raise ValueError("The sum of train_fraction, val_fraction, and test_fraction must equal 1.0")

        if mask_paths_multilabel != None and len(image_files) != len(mask_paths_multilabel):
            raise ValueError("The number of images and masks must be the same.")
        
        if hold_out: 
            num_classes = None
            if stratify:
                train_images, train_masks, val_images, val_masks, test_images, test_masks, num_classes = stratify_split(image_files, mask_paths_multilabel, train_fraction, val_fraction, test_fraction, stratification_strategy, background, random_state)
            else:
                train_images, train_masks, val_images, val_masks, test_images, test_masks = random_split(image_files, mask_paths_multilabel, train_fraction, val_fraction, test_fraction, random_state)
        
            train_df = pd.DataFrame({"img_path": train_images, "mask_path": train_masks})
            val_df = pd.DataFrame({"img_path": val_images, "mask_path": val_masks})
            test_df = pd.DataFrame({"img_path": test_images, "mask_path": test_masks})

            save_split_to_directory(train_df, val_df, test_df, output_dir)

            return num_classes
    

    def check_partitions_in_directories(output_dir: str, total_images:int, num_classes: int, test_fraction: float, n_splits: int):
        """
        Verifica la correcta partición y la distribución de clases en los directorios de salida generados por `automate_split_nested`.
        
        Args:
            output_dir: Directorio principal donde se guardan las particiones (test y folds).
            num_classes: Número total de clases en las máscaras.
            test_fraction: Fracción del dataset reservado para el conjunto de test.
            n_splits: Número de folds de validación cruzada.
        """
        # Inicializar los contadores de clases
        class_counts = {split: {cls: 0 for cls in range(num_classes)} for split in ['test'] + [f'fold_{i+1}' for i in range(n_splits)]}

        # 1. Verificar el número de archivos en el directorio 'test'
        test_img_dir = os.path.join(output_dir, 'test', 'images')
        test_label_dir = os.path.join(output_dir, 'test', 'labels')
        
        if not os.path.exists(test_img_dir) or not os.path.exists(test_label_dir):
            raise ValueError(f"Directorio no encontrado: {test_img_dir} o {test_label_dir}")
        
        test_img_files = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
        test_label_files = [f for f in os.listdir(test_label_dir) if f.endswith('.png')]

        if len(test_img_files) != len(test_label_files):
            raise ValueError(f"El número de imágenes y máscaras no coincide en el directorio de 'test'")

        print(f"Test - Imágenes: {len(test_img_files)}, Máscaras: {len(test_label_files)}")

        # 2. Comprobar la distribución de clases en el directorio de test
        for mask_file in test_label_files:
            mask = np.array(Image.open(os.path.join(test_label_dir, mask_file)))
            unique, counts = np.unique(mask, return_counts=True)
            for cls in unique:
                class_counts['test'][cls] += 1

        # 3. Comprobar la distribución de clases en los directorios de los folds
        for fold_idx in range(n_splits):
            fold_img_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}", 'train', 'images')
            fold_label_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}", 'train', 'labels')
            fold_img_dir_val = os.path.join(output_dir, f"fold_{fold_idx + 1}", 'val', 'images')
            fold_label_dir_val = os.path.join(output_dir, f"fold_{fold_idx + 1}", 'val', 'labels')

            
            if not os.path.exists(fold_img_dir) or not os.path.exists(fold_label_dir):
                raise ValueError(f"Directorio no encontrado para el fold {fold_idx + 1}: {fold_img_dir} o {fold_label_dir}")
            
            fold_img_files = [f for f in os.listdir(fold_img_dir) if f.endswith('.jpg')]
            fold_label_files = [f for f in os.listdir(fold_label_dir) if f.endswith('.png')]
            fold_img_dir_val = [f for f in os.listdir(fold_img_dir_val) if f.endswith('.jpg')]
            fold_label_dir_val = [f for f in os.listdir(fold_label_dir_val) if f.endswith('.png')]

            if len(fold_img_files) != len(fold_label_files):
                raise ValueError(f"El número de imágenes y máscaras no coincide en el directorio de 'fold_{fold_idx + 1}'")

            print(f"Fold {fold_idx + 1} - Imágenes (Train): {len(fold_img_files)}, Máscaras (Train): {len(fold_label_files)} | Imágenes (Val): {len(fold_img_dir_val)}, Máscaras (Val): {len(fold_label_dir_val)}")

            # 4. Comprobar la distribución de clases en los folds
            for mask_file in fold_label_files:
                mask = np.array(Image.open(os.path.join(fold_label_dir, mask_file)))
                unique, counts = np.unique(mask, return_counts=True)
                for cls in unique:
                    class_counts[f'fold_{fold_idx + 1}'][cls] += 1

        # 5. Mostrar la distribución de clases en cada conjunto
        print("\nDistribución de clases en 'test' y 'folds':")
        for split in ['test'] + [f'fold_{i+1}' for i in range(n_splits)]:
            print(f"\nDistribución de clases en {split}:")
            for cls in range(num_classes):
                print(f"  Clase {cls}: {class_counts[split][cls]} imágenes")

        # 6. Comprobar la fracción de test
        test_images = len([f for f in os.listdir(os.path.join(output_dir, 'test', 'images')) if f.endswith('.jpg')])

        expected_test_size = total_images * test_fraction
        if not np.isclose(test_images, expected_test_size, atol=1):
            raise ValueError(f"El número de imágenes en el conjunto de test no coincide con la fracción esperada. "
                            f"Esperado: {expected_test_size}, encontrado: {test_images}")
        else:
            print(f"[OK] La fracción de imágenes de test es correcta. Se esperaba: {expected_test_size}, encontrado: {test_images}")

        print("[OK] Todos los directorios y particiones verificadas correctamente.")

    img_path = r"C:\Users\joortif\Desktop\datasets\Preprocesados\forest_freiburg_multiclass\full\images"
    label_path = r"C:\Users\joortif\Desktop\datasets\Preprocesados\forest_freiburg_multiclass\full\labels"

    #img_path = r"C:\Users\joortif\Desktop\datasets\Preprocesados\drone_dataset_reduc\full\images"
    #label_path = r"C:\Users\joortif\Desktop\datasets\Preprocesados\drone_dataset_reduc\full\labels"

    num_classes = 6

    image_files = sorted([os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.jpg')])
    label_files = sorted([os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith('.png')])

    output_dir = r"C:\Users\joortif\Desktop\Resultados_ImageDatasetAnalyzer\dataset_partitions"

    lbldataset = ImageLabelDataset(img_path, label_path)

    #lbldataset.analyze(plot=False, output=None, verbose=True)

    split(image_files=image_files, mask_paths_multilabel=label_files, num_classes=num_classes, train_fraction=0.7, val_fraction=0.2, test_fraction=0.1, hold_out=True, stratify=False, stratification_strategy="pixel_to_object_ratio", background=None, output_dir=output_dir, random_state=123)
    
    #check_partitions_in_directories(output_dir=output_dir, total_images = 343, num_classes=num_classes, test_fraction=0.1, n_splits=5)

    #check_objects_per_class(output_dir, class_names=['Obstacle', 'Water', 'Nature', 'Water','Moving'], dataset_name="drone", num_classes=5)

    #check_class_distribution(output_dir, "forest_strat", num_classes)

    check_ratio_per_class(output_dir, ['Cielo', 'Obstaculo', 'Vacio', 'Cesped','Vegetacion', 'Camino'], "forest")