from src.extensions.extensions import LabelExtensions
from src.models.depthestimator import DepthEstimator
from src.utils.utils import Utils
from src.utils.imagelabelutils import ImageLabelUtils
from src.transformers.transformer import Transformer

import cv2
import os 
import numpy as np
from tqdm import tqdm

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

    def transform(self, input_data: str, output_dir: str, img_path: str, fill_background: int | None, depth_model: str ="Intel/dpt-swinv2-tiny-256"):
        
        os.makedirs(output_dir, exist_ok=True)

        converted_masks = []
        labels = os.listdir(input_data)
        device = Utils.get_device(self.logger)
        depth_estimator = DepthEstimator(depth_model, device)

        for label_name in tqdm(labels, desc="Converting labels from TXT YOLO format to multilabel..."):

            label_path = os.path.join(input_data, label_name)

            objects = self._read_yolo(label_path)

            image_path = ImageLabelUtils.label_to_image(label_path, img_path, LabelExtensions.JPG.value)

            depth_map = depth_estimator.generate_depth_map(image_path)

            h, w = depth_map.shape
            mask = self._create_empty_mask(h, w, fill_background)

            object_depths = []

            for class_id, points in objects:
                mask_obj = np.zeros_like(depth_map, dtype=np.uint8)

                points = self._scale_polygon(points, h, w)

                cv2.fillPoly(mask_obj, [points.astype(np.int32)], 255)
                
                depth_values = depth_map[mask_obj == 255]
                
                if depth_values.size > 0:
                    depth_mean = np.mean(depth_values)
                    object_depths.append((depth_mean, class_id, points))

            object_depths.sort(key=lambda x: x[0], reverse=True)

            for _, class_id, points in object_depths:
                cv2.fillPoly(mask, [points.astype(np.int32)], class_id)

            converted_masks.append(mask)

            ImageLabelUtils.save_multilabel_mask(mask, label_name, output_dir)
            
        return converted_masks
