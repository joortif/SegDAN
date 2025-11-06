import os
from typing import Tuple
from torch.utils.data import Dataset as BaseDataset
import os
import cv2
import numpy as np

from segdan.utils.confighandler import ConfigHandler

class SemanticSegmentationDataset(BaseDataset):
    
    def __init__(self, classes, images_dir=None, masks_dir=None, image_paths=None, mask_paths=None, augmentation=None, background=None, binary=True):
        
        self.augmentation = augmentation
        self.classes = classes
        self.binary = binary
        self.background_class = background

        if image_paths is not None and mask_paths is not None:
            assert len(image_paths) == len(mask_paths)
            self.image_paths = image_paths
            self.mask_paths = mask_paths
            self.mode = "paths"
            self.ids = [os.path.basename(p) for p in image_paths]
        else:
            self.mode = "dirs"
            self.image_paths = []
            self.mask_paths = []
            self.ids = []

            for fname in os.listdir(images_dir):
                if fname.lower().endswith(tuple(ConfigHandler.VALID_IMAGE_EXTENSIONS)):
                    img_path = os.path.join(images_dir, fname)
                    mask_name = os.path.splitext(fname)[0] + ".png"
                    mask_path = os.path.join(masks_dir, mask_name)

                    if os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                        self.ids.append(fname)
        
        self.class_values = [self.classes.index(cls.lower()) for cls in classes]
            
        # Create a remapping dictionary: class value in dataset -> new index (0, 1, 2, ...)
        # Background will always be 255, other classes will be remapped starting from 1.
        
        if self.binary:
            self.class_map = {v: 0 if v == self.background_class else 1 for v in self.class_values}
        else:
            self.class_map = {self.background_class: 255} if self.background_class is not None else {}
            self.class_map.update({v: i for i, v in enumerate(self.class_values) if v != self.background_class})
            
        self.augmentation = augmentation

    def __getitem__(self, i):
        return self._get_raw(i)

    def _get_raw(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.image_paths[i]
        mask_path = self.mask_paths[i]

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise RuntimeError(f"Image read failed: {img_path}")

        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise RuntimeError(f"Mask read failed: {mask_path}")

        if self.binary:
            mask_remap = np.where(mask == 0, 0, 1).astype(np.uint8)
        else:
            mask_remap = np.full_like(mask, 255 if self.background_class is not None else 0, dtype=np.uint8)
            for class_value, new_value in self.class_map.items():
                mask_remap[mask == class_value] = new_value

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]

        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = (image * 255).round().astype(np.uint8) if image.max() <= 1.0 else image.round().astype(np.uint8)

        mask_remap = np.asarray(mask_remap).astype(np.int64)

        return image, mask_remap
    
    def __len__(self): 
        return len(self.ids)
