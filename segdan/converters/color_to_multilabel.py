from typing import Optional
import numpy as np
import cv2
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

from segdan.converters.converter import Converter
from segdan.utils.imagelabelutils import ImageLabelUtils

logger = logging.getLogger(__name__)

class ColorToMultilabelConverter(Converter):

    num_images_mosaic = 4

    def __init__(self, input_data: str, output_dir: str, color_dict: Optional[dict] = None):
        super().__init__(input_data, output_dir)
        self.color_dict = color_dict

    def generate_color_dict(self):
        label_files = [f for f in os.listdir(self.input_data) if os.path.isfile(os.path.join(self.input_data, f))]
        unique_colors = set()

        for filename in tqdm(label_files, desc="Generating unique colors..."):
            label_path = os.path.join(self.input_data, filename)
            color_mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            colors = np.unique(color_mask.reshape(-1, color_mask.shape[-1]), axis=0)
            unique_colors.update(map(tuple, colors))

        assigned_colors = {}
        image_dir = self.input_data.replace('labels', 'images')
        
        for color in unique_colors:

            valid_files = []

            for filename in label_files:
                label_path = os.path.join(self.input_data, filename)
                mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
                if np.any(np.all(mask.reshape(-1, mask.shape[-1]) == color, axis=1)):
                    valid_files.append(filename)

            selected_files = random.sample(valid_files, min(self.num_images_mosaic, len(valid_files)))
            fig, axes = plt.subplots(1, self.num_images_mosaic, figsize=(15, 5))
            
            for i, filename in enumerate(selected_files):
                label_path = os.path.join(self.input_data, filename)
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
                    logger.error(f"Invalid input. Class ID already assigned.")
                except ValueError:
                    logger.error("Invalid input. Please enter a valid integer ID.")
            plt.close()  
            
        return assigned_colors

    def convert(self):
        masks = []
        os.makedirs(self.output_dir, exist_ok=True)

        label_files = [f for f in os.listdir(self.input_data) if os.path.isfile(os.path.join(self.input_data, f))]

        if self.color_dict is None:
            logger.info("No color dictionary found. It will be created automatically...")
            self.color_dict = self.generate_color_dict(self.input_data)

        for filename in tqdm(label_files, desc="Converting labels from color format to multilabel..."):
            label_path = os.path.join(self.input_data, filename)
            color_mask = cv2.imread(label_path)
            multilabel_mask = np.zeros(color_mask.shape[:2], dtype=np.uint8)
            pixel_colors = color_mask.reshape(-1, color_mask.shape[-1])

            for idx, pixel_color in enumerate(pixel_colors):
                color_tuple = tuple(pixel_color)
                if color_tuple in self.color_dict:
                    multilabel_mask.reshape(-1)[idx] = self.color_dict[color_tuple]

            multilabel_mask = multilabel_mask.reshape(color_mask.shape[:2])
            masks.append(multilabel_mask)

            ImageLabelUtils.save_multilabel_mask(multilabel_mask, filename, self.output_dir)

        return masks