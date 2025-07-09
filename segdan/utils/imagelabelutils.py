import os
from typing import Optional
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

from segdan.exceptions.exceptions import ExtensionNotFoundException
from segdan.extensions.extensions import LabelExtensions
from segdan.utils.confighandler import ConfigHandler

class ImageLabelUtils:

    @staticmethod
    def label_to_image(label_path: str, img_path:str , img_ext:str ) -> str: 

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Directory {label_path} does not exist.")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Directory {img_path} does not exist.")

        label_name, _ = os.path.splitext(os.path.basename(label_path))

        image = os.path.join(img_path, f"{label_name}{img_ext}")

        return image
    
    @staticmethod
    def image_to_label(img_path: str, label_path:str , lbl_ext:str ) -> str: 

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Directory {img_path} does not exist.")

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Directory {label_path} does not exist.")
        
        image_name, _ = os.path.splitext(os.path.basename(img_path))

        label = os.path.join(label_path, f"{image_name}{lbl_ext}")

        return label
    
    @staticmethod
    def save_multilabel_mask(mask, file_name, output_dir: str):
        mask_filename = os.path.splitext(file_name)[0] + ".png"
        mask_filepath = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_filepath, mask)

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
    def check_label_extensions(label_dir, verbose=False, logger=None):
        """
        Checks the extensions of label files in the label directory to ensure consistency.

        Args:
            verbose (bool): If True, logs the process and any issues found.

        Returns:
            str: The extension of the label files if consistent.

        Raises:
            ValueError: If multiple extensions are found in the label directory.
            ExtensionNotFoundException: If the extension is not recognized by the system.
        """
        
        if verbose:
            logger.info(f"Checking label extensions from path: {label_dir}...")

        labels_ext = {os.path.splitext(file)[1].lower() for file in os.listdir(label_dir) if os.path.splitext(file)[1].lower() in VALID_ANNOTATION_EXTENSIONS}

        if len(labels_ext) == 1:
            ext = labels_ext.pop()  
            try:
                enum_ext = LabelExtensions.extensionToEnum(ext)

                if enum_ext:
                    if verbose:
                        logger.info(f"All labels are in {enum_ext.name} format.")

                    return LabelExtensions.enumToExtension(enum_ext)
                
            except ExtensionNotFoundException as e:
                print(f"All labels are in unknown {ext} format.")
                raise e
        else:
            raise ValueError(f"The directory contains multiple extensions for labels: {labels_ext}.")
        
    @staticmethod
    def all_images_are_color(directory):
        for filename in os.listdir(directory):
            image_path = os.path.join(directory, filename)
            if not os.path.isfile(image_path):
                continue

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                continue  

            if len(image.shape) == 2:
                return False

            if image.shape[2] >= 3:
                b, g, r = image[..., 0], image[..., 1], image[..., 2]
                if np.array_equal(b, g) and np.array_equal(b, r):
                    return False  

        return True
    
    @staticmethod
    def get_classes_from_csv(csv_file, background: Optional[int]=None):
        df = pd.read_csv(csv_file, delimiter=";")

        class_names_dict = {i: str(class_name) for i, class_name in enumerate(df["Class name"].values)}

        if background:
            class_names_dict[background] = "background"

        return list(class_names_dict.values())

    @staticmethod
    def count_num_classes_png(label_dir):
        all_classes = set()

        for filename in tqdm(os.listdir(label_dir), desc="Counting number of classes from multiclass labels..."):
            if filename.endswith('.png'):
                path = os.path.join(label_dir, filename)
                mask = np.array(Image.open(path))
                unique_classes = np.unique(mask)
                all_classes.update(unique_classes)

        return len(all_classes), all_classes