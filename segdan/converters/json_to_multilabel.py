import json
import os
from tqdm import tqdm
import cv2
import numpy as np
import logging

from typing import Optional
from segdan.models.depthestimator import DepthEstimator
from segdan.utils.imagelabelutils import ImageLabelUtils
from segdan.utils.utils import Utils
from segdan.converters.converter import Converter

logger = logging.getLogger(__name__)

class JSONToMultilabelConverter(Converter):

    def __init__(self, input_data: str, output_dir: str, img_dir: str, background: Optional[int], depth_model: str ="Intel/dpt-swinv2-tiny-256"):
        super().__init__(input_data, output_dir)

        self.img_dir = img_dir
        self.fill_background = background
        self.depth_model = depth_model

    def _read_json(self, json_path: str):
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)

    def convert(self):

        os.makedirs(self.output_dir, exist_ok=True)

        transformed_masks = []

        data = self._read_json(self.input_data)
        device = Utils.get_device()
        depth_estimator = DepthEstimator(self.depth_model, device)

        for img_info in tqdm(data['images'], desc="Converting labels from JSON COCO format to multilabel..."):
            img_id = img_info['id']
            img_filename = img_info['file_name']

            image_path = os.path.join(self.img_dir, img_filename)

            if not os.path.exists(image_path):
                logger.info(f"Image {img_filename} not found, skipping...")
                continue

            depth_map = depth_estimator.generate_depth_map(image_path)

            h, w = depth_map.shape
            mask = self._create_empty_mask(h, w, self.fill_background)

            annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] == img_id]

            object_depths = []

            for ann in annotations:
                mask_obj = np.zeros_like(depth_map, dtype=np.uint8)
                category_id = ann['category_id']
                segmentation = ann['segmentation']

                for polygon in segmentation:
                    polygon = np.array(polygon, dtype=np.float32).reshape(-1, 2)

                    if np.max(polygon) <= 1:
                        polygon = self._scale_polygon(polygon, h, w)
                    
                    cv2.fillPoly(mask_obj, [polygon.astype(np.int32)], 255)  

                    depth_values = depth_map[mask_obj == 255]
                    if depth_values.size > 0:
                        depth_mean = np.mean(depth_values)

                        object_depths.append((depth_mean, category_id, polygon))

            object_depths.sort(key=lambda x: x[0], reverse=True)

            for _, category_id, polygon in object_depths:
                cv2.fillPoly(mask, [polygon.astype(np.int32)], category_id)  

            transformed_masks.append(mask)
                
            ImageLabelUtils.save_multilabel_mask(mask, img_filename, self.output_dir)

        return transformed_masks
