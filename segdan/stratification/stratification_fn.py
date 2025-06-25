import cv2
from PIL import Image
import numpy as np

from tqdm import tqdm

def calculate_pixel_distribution(mask_paths, background: int | None = None):
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

def calculate_object_number(masks, background: int | None = None):
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
    
def calculate_pixel_ratio(masks, background: int | None = None):
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