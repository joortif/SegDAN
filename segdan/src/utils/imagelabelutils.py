import os
import cv2

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

