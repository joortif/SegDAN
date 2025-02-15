import os
import cv2
import numpy as np
from PIL import Image

from src.exceptions.exceptions import ExtensionNotFoundException
from src.extensions.extensions import LabelExtensions

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

        labels_ext = {os.path.splitext(file)[1] for file in os.listdir(label_dir)}

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
            image = Image.open(image_path)
            img_array = np.array(image)
                
            if not (img_array.ndim == 3 and img_array.shape[2] == 3): 
                return False

        return True
