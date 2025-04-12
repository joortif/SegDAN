from src.transformers.transformer import Transformer
from src.utils.imagelabelutils import ImageLabelUtils

import numpy as np
import gc
import os
import cv2
from tqdm import tqdm

class BinaryToMultilabelTransformer(Transformer):

    def __init__(self):
        super().__init__()
        self.MAX_SIZE = 4000

    """def transform(self, input_data: str, img_dir: str, output_dir: str, threshold: int = 255):
        masks = []

        label_files = [f for f in os.listdir(input_data) if os.path.isfile(os.path.join(input_data, f))]

        os.makedirs(output_dir, exist_ok=True)

        for filename in tqdm(label_files, desc="Converting labels from binary format to multilabel..."):
            label_path = os.path.join(input_data, filename)

            mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            height, width = mask.shape

            original_height, original_width = height, width

            if max(height, width) > self.MAX_SIZE:
                scale = self.MAX_SIZE / max(height, width)

                if max(height, width) > 2 * self.MAX_SIZE:
                    mask = cv2.imread(label_path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
                    height, width = mask.shape  
                    scale = self.MAX_SIZE / max(height, width)  
                
                new_size = (int(width * scale), int(height * scale))
                mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_AREA)

                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir BGR a RGB

                # Redimensionar la imagen al nuevo tamaño (lo mismo que la máscara)
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

            converted_mask = (mask >= threshold).astype(np.uint8)  

            ImageLabelUtils.save_multilabel_mask(converted_mask, filename, output_dir)
            masks.append(converted_mask)

            img_path = os.path.join(img_dir, filename.replace('.png', '.jpg')) 
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

            if (original_height, original_width) != new_size:
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
                image_output_path = os.path.join(img_dir, filename.replace('.png', '.jpg'))  
                cv2.imwrite(image_output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            del mask, converted_mask             
            gc.collect()
            
        return masks"""
    
    def transform(self, input_data: str, img_dir: str, output_dir: str, threshold: int = 255):
        masks = []

        label_files = [f for f in os.listdir(input_data) if os.path.isfile(os.path.join(input_data, f))]

        os.makedirs(output_dir, exist_ok=True)

        for filename in tqdm(label_files, desc="Converting labels from binary format to multilabel..."):
            label_path = os.path.join(input_data, filename)
            
            mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            converted_mask = (mask >= threshold).astype(np.uint8)

            ImageLabelUtils.save_multilabel_mask(converted_mask, filename, output_dir)
            masks.append(converted_mask)

            del mask, converted_mask
            gc.collect()

        return masks

    


