from src.models.depthestimator import DepthEstimator
from src.utils.utils import Utils
from src.utils.imagelabelutils import ImageLabelUtils
from src.transformers.transformer import Transformer

import cv2
import os 
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


class YOLOToMultilabelTransformer(Transformer):

    def __init__(self):
        super().__init__()
        
    def _read_yolo(self, yolo_path: str):
            annotations = []
            with open(yolo_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    class_id = int(values[0])
                    values = np.array(list(map(float, values[1:]))).reshape(-1, 2)
                    annotations.append((class_id,values))

            return annotations

    def transform(self, input_data: str, img_path: str, img_ext: str, fill_background: int | None, depth_model: str ="Intel/dpt-swinv2-tiny-256", output_path: str = None, verbose: bool = False):
        
        os.makedirs(output_path, exist_ok=True)

        converted_masks = []
        labels = os.listdir(input_data)
        device = Utils.get_device(self.logger)
        depth_estimator = DepthEstimator(depth_model, device)

        for label_name in tqdm(labels, desc="Converting labels from TXT YOLO format to multilabel..."):

            label_path = os.path.join(input_data, label_name)

            objects = self._read_yolo(label_path)

            image_path = ImageLabelUtils.label_to_image(label_path, img_path, img_ext)

            depth_map = depth_estimator.generate_depth_map(image_path)

            h, w = depth_map.shape
            mask = self._create_empty_mask(h, w, fill_background)

            object_depths = []

            for class_id, points in objects:
                mask_obj = np.zeros_like(depth_map, dtype=np.uint8)

                points = self._scale_polygon(points, h, w)

                cv2.fillPoly(mask_obj, [points.astype(np.int32)], 255)
                
                #plt.imshow(mask_obj, cmap='gray')
                #plt.title(f'Máscara del objeto clase {class_id}')
                #plt.axis('off')
                #plt.show()
                
                depth_values = depth_map[mask_obj == 255]

                depth_mean = np.mean(depth_values)

                object_depths.append((depth_mean, class_id, points))

            object_depths.sort(key=lambda x: x[0], reverse=True)

            for _, class_id, points in object_depths:
                cv2.fillPoly(mask, [points.astype(np.int32)], class_id)

            converted_masks.append(mask)

            ImageLabelUtils.save_multilabel_mask(mask, label_name, output_path)

            Utils.overlay_mask_on_image(image_path, mask)
            
        return converted_masks
