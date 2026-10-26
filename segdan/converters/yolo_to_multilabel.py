from pathlib import Path
from typing import Optional
import logging
import os

import cv2
import numpy as np
from tqdm import tqdm

from segdan.models.depthestimator import DepthEstimator
from segdan.utils.utils import Utils
from segdan.utils.imagelabelutils import ImageLabelUtils
from segdan.converters.converter import Converter
from segdan.utils.confighandler import ConfigHandler

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ConfigHandler.VALID_IMAGE_EXTENSIONS

class YOLOToMultilabelConverter(Converter):

    def __init__(
        self,
        input_data: str,
        output_dir: str,
        img_dir: str,
        background: Optional[int],
        depth_model: str = "Intel/dpt-swinv2-tiny-256",
    ):
        super().__init__(input_data, output_dir)
        self.img_dir = img_dir
        self.fill_background = background
        self.depth_model = depth_model

    @staticmethod
    def _read_yolo(yolo_path: Path):
        annotations = []
        with yolo_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue

                values = line.split()
                try:
                    class_id = int(values[0])
                    coordinates = np.asarray(values[1:], dtype=np.float32)
                except (ValueError, IndexError) as exc:
                    raise ValueError(
                        f"Invalid YOLO annotation at {yolo_path}:{line_no}: {line!r}"
                    ) from exc

                if class_id < 0:
                    raise ValueError(
                        f"Negative class ID at {yolo_path}:{line_no}: {class_id}"
                    )
                if coordinates.size < 6 or coordinates.size % 2:
                    raise ValueError(
                        f"Invalid segmentation polygon at {yolo_path}:{line_no}. "
                        "Expected a class ID followed by at least three normalized x/y points. "
                        "Detection bounding boxes are not supported."
                    )

                points = coordinates.reshape(-1, 2)
                annotations.append((class_id, points))

        return annotations

    @staticmethod
    def _index_by_stem(paths, description: str):
        indexed = {}
        duplicates = []
        for path in paths:
            key = path.stem.casefold()
            if key in indexed:
                duplicates.append((indexed[key], path))
            else:
                indexed[key] = path

        if duplicates:
            examples = ", ".join(
                f"{first.name} / {second.name}" for first, second in duplicates[:5]
            )
            raise ValueError(
                f"Multiple {description} files share the same base name: {examples}. "
                "Rename them so each image has an unambiguous annotation match."
            )
        return indexed

    def _get_image_paths(self):
        image_root = Path(self.img_dir)
        if not image_root.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {image_root}")

        image_paths = sorted(
            p for p in image_root.iterdir()
            if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(
                f"No supported images found in {image_root}. Supported extensions: "
                f"{', '.join(sorted(IMAGE_EXTENSIONS))}"
            )
        return image_paths

    def _get_label_paths(self):
        label_root = Path(self.input_data)
        if label_root.is_file() and label_root.suffix.casefold() == ".txt":
            return [label_root]
        if not label_root.is_dir():
            raise FileNotFoundError(f"YOLO labels path does not exist: {label_root}")

        label_paths = sorted(
            p for p in label_root.iterdir()
            if p.is_file() and p.suffix.casefold() == ".txt"
        )
        if not label_paths:
            raise ValueError(f"No YOLO .txt labels found in {label_root}")
        return label_paths

    def convert(self):
        image_paths = self._get_image_paths()
        label_paths = self._get_label_paths()

        image_by_stem = self._index_by_stem(image_paths, "image")
        label_by_stem = self._index_by_stem(label_paths, "label")

        orphan_labels = sorted(
            label_by_stem[key].name
            for key in set(label_by_stem) - set(image_by_stem)
        )
        if orphan_labels:
            raise ValueError(
                f"YOLO labels have no matching image in {self.img_dir} "
                f"(first 10): {orphan_labels[:10]}"
            )

        os.makedirs(self.output_dir, exist_ok=True)
        converted_masks = []
        device = Utils.get_device()
        depth_estimator = DepthEstimator(self.depth_model, device)

        background = 0 if self.fill_background is None else int(self.fill_background)
        if not 0 <= background <= 255:
            raise ValueError(
                f"The mask background must fit in uint8 (0-255), got {background}."
            )

        for image_path in tqdm(image_paths, desc="Converting YOLO TXT labels to multilabel masks"):
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(
                    f"OpenCV could not decode image {image_path}. "
                    "Check that the image format is supported by your OpenCV build."
                )
            height, width = image.shape[:2]
            del image

            label_path = label_by_stem.get(image_path.stem.casefold())
            objects = self._read_yolo(label_path) if label_path else []

            if objects:
                depth_map = depth_estimator.generate_depth_map(str(image_path))
                height, width = depth_map.shape[:2]
                object_depths = []

                for class_id, normalized_points in objects:
                    points = normalized_points.copy()
                    points[:, 0] = np.clip(points[:, 0] * width, 0, width - 1)
                    points[:, 1] = np.clip(points[:, 1] * height, 0, height - 1)
                    points = np.rint(points).astype(np.int32)

                    mask_obj = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(mask_obj, [points], 255)
                    depth_values = depth_map[mask_obj == 255]
                    if depth_values.size:
                        object_depths.append((float(np.mean(depth_values)), class_id, points))

                object_depths.sort(key=lambda item: item[0], reverse=True)
                mask = np.full((height, width), background, dtype=np.uint8)
                for _, class_id, points in object_depths:
                    if not 0 <= class_id <= 255:
                        raise ValueError(
                            f"Class ID {class_id} from {label_path} cannot be stored in an 8-bit mask."
                        )
                    cv2.fillPoly(mask, [points], int(class_id))
            else:
                mask = np.full((height, width), background, dtype=np.uint8)

            ImageLabelUtils.save_multilabel_mask(mask, image_path.name, self.output_dir)
            converted_masks.append(mask)

        logger.info(
            "Converted %d images into PNG masks in %s (%d images had no matching TXT and were "
            "saved as background-only masks).",
            len(image_paths),
            self.output_dir,
            sum(image_path.stem.casefold() not in label_by_stem for image_path in image_paths),
        )
        return converted_masks