import os
from torch.utils.data import Dataset as BaseDataset
import os
import cv2
import numpy as np


class SemanticSegmentationDataset(BaseDataset):
    
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, background=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        
        masks_ids = [os.path.splitext(image_id)[0]+'.png' for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in masks_ids]
        self.background_class = background
        
        self.classes = classes
        if classes:
            self.class_values = [self.classes.index(cls.lower()) for cls in classes]
        else:
            self.class_values = list(range(len(self.classes)))
            
        # Create a remapping dictionary: class value in dataset -> new index (0, 1, 2, ...)
        # Background will always be 255, other classes will be remapped starting from 1.
        self.class_map = {self.background_class: 255} if self.background_class is not None else {}
        self.class_map.update(
            {
                v: i
                for i, v in enumerate(self.class_values)
                if v != self.background_class
            }
        )
                
        self.augmentation = augmentation

    def __getitem__(self, i):
        # Read the image
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Read the mask in grayscale mode
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # Create a blank mask to remap the class values
        mask_remap = np.full_like(mask, 255 if self.background_class is not None else 0, dtype=np.uint8)

        # Remap the mask according to the dynamically created class map
        for class_value, new_value in self.class_map.items():
            mask_remap[mask == class_value] = new_value
            
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]
        image = image.transpose(2, 0, 1)
        return image, mask_remap

    def __len__(self):
        return len(self.ids)
    
    def get_class_map(self):
        return self.class_map