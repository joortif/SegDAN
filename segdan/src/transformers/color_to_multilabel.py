from src.utils.imagelabelutils import ImageLabelUtils
from src.transformers.transformer import Transformer

import numpy as np
import cv2
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class ColorToMultilabelTransformer(Transformer):

    num_images_mosaic = 4

    def __init__(self):
        super().__init__()

    def generate_color_dict(self, input_data: str):
        label_files = [f for f in os.listdir(input_data) if os.path.isfile(os.path.join(input_data, f))]
        unique_colors = set()

        for filename in tqdm(label_files, desc="Generating unique colors..."):
            label_path = os.path.join(input_data, filename)
            color_mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            colors = np.unique(color_mask.reshape(-1, color_mask.shape[-1]), axis=0)
            unique_colors.update(map(tuple, colors))

        assigned_colors = {}
        image_dir = input_data.replace('labels', 'images')
        
        for color in unique_colors:

            valid_files = []

            for filename in label_files:
                label_path = os.path.join(input_data, filename)
                mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
                if np.any(np.all(mask.reshape(-1, mask.shape[-1]) == color, axis=1)):
                    valid_files.append(filename)

            selected_files = random.sample(valid_files, min(self.num_images_mosaic, len(valid_files)))
            fig, axes = plt.subplots(1, self.num_images_mosaic, figsize=(15, 5))
            
            for i, filename in enumerate(selected_files):
                label_path = os.path.join(input_data, filename)
                image_path = ImageLabelUtils.label_to_image(label_path, image_dir, '.jpg')

                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
                mask_binary = np.all(mask == color, axis=-1).astype(np.uint8)
                mask_colored = np.zeros_like(img)
                mask_colored[mask_binary == 1] = color

                overlay = cv2.addWeighted(img, 0.5, mask_colored, 1, 0)
                axes[i].imshow(overlay)
                axes[i].set_title(filename)
                axes[i].axis("off")
            
            plt.show(block=False)  
            while True:
                try:
                    color_id = int(input(f"Enter class ID for color {color}: "))
                    if color_id not in assigned_colors.values():
                        assigned_colors[color] = color_id
                        break
                    print(f"Invalid input. Class ID already assigned.")
                except ValueError:
                    print("Invalid input. Please enter a valid integer ID.")
            plt.close()  
            
        return assigned_colors

    def transform(self, input_data: str, output_dir: str, color_dict: dict | None = None):
        masks = []
        os.makedirs(output_dir, exist_ok=True)

        label_files = [f for f in os.listdir(input_data) if os.path.isfile(os.path.join(input_data, f))]

        if color_dict is None:
            print("No color dictionary found. It will be created automatically...")
            color_dict = self.generate_color_dict(input_data)

        for filename in tqdm(label_files, desc="Converting labels from color format to multilabel..."):
            label_path = os.path.join(input_data, filename)
            color_mask = cv2.imread(label_path)
            multilabel_mask = np.zeros(color_mask.shape[:2], dtype=np.uint8)
            pixel_colors = color_mask.reshape(-1, color_mask.shape[-1])

            for idx, pixel_color in enumerate(pixel_colors):
                color_tuple = tuple(pixel_color)
                if color_tuple in color_dict:
                    multilabel_mask.reshape(-1)[idx] = color_dict[color_tuple]

            multilabel_mask = multilabel_mask.reshape(color_mask.shape[:2])
            masks.append(multilabel_mask)

            ImageLabelUtils.save_multilabel_mask(multilabel_mask, filename, output_dir)

        return masks