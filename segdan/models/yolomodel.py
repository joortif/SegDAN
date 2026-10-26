import logging
import time 
from pathlib import Path

import yaml
from segdan.datasets.augments import get_yolo_augmentation
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset


logger = logging.getLogger(__name__)

class YOLOModel:

    SIZE_MAP = {
        "nano": "n",
        "small": "s",
        "medium": "m",
        "large": "l",
        "extra-large": "x",
        "n": "n",
        "s": "s",
        "m": "m",
        "l": "l",
        "x": "x"
    }

    def __init__(self, classes, epochs, imgsz, metrics, selection_metric, output_path, 
                 model_name="yolo", model_size="small", batch_size=8, patience=20, device=None, workers=4, seed=42):

        family = str(model_name).lower().replace("-", "")

        if family not in {"yolo", "yolo11", "yolo11seg"}:
            raise ValueError(f"Invalid model name: {model_name}. Supported models are: yolo, yolo11, yolo11seg")

        size = str(model_size).lower() 
        if size not in self.SIZE_MAP:
            raise ValueError(f"Invalid model size: {model_size}. Supported sizes are: {list(self.SIZE_MAP.keys())}")

        self.model_name = "yolo11"
        self.model_size = self.SIZE_MAP[size]

        self.weights_name = f"yolo11{self.model_size}-seg.pt"
        self.classes = [
            str(cls)
            for cls in classes
            if str(cls).lower() != "background"
        ]

        self.epochs = int(epochs)
        self.imgsz = int(imgsz)
        self.metrics = list(metrics)
        self.selection_metric = (
            str(selection_metric).strip() or "map"
        )
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.device = device
        self.workers = int(workers)
        self.seed = int(seed)

        self.output_path = Path(output_path).resolve()
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.data_yaml = None
        self.has_test = False
        self.best_weights = None
        self.validation_metrics = {}
        self.training_time = None

    def _validate_labels(self, split_dir):
            labels_dir = Path(split_dir) / "labels"

            for label_path in labels_dir.glob("*.txt"):
                for line_number, line in enumerate(
                    label_path.read_text(encoding="utf-8").splitlines(),
                    start=1,
                ):
                    if not line.strip():
                        continue

                    values = line.split()

                    if len(values) < 7 or (len(values) - 1) % 2 != 0:
                        raise ValueError(
                            f"Bad YOLO label format in "
                            f"{label_path}:{line_number}"
                        )

                    try:
                        class_id = int(values[0])
                        coords = [float(value) for value in values[1:]]
                    except ValueError as exc:
                        raise ValueError(
                            f"Bad YOLO label format in "
                            f"{label_path}:{line_number}"
                        ) from exc

                    if not 0 <= class_id < len(self.classes):
                        raise ValueError(
                            f"Class {class_id} out of range {label_path}. "
                            f"Expected IDs between 0 and {len(self.classes) - 1}."
                        )

                    if not all(
                        0.0 <= value <= 1.0 for value in coords
                    ):
                        raise ValueError(
                            f"Coordinates not normalized in {label_path}."
                        )

    def create_dataset_yaml(self, split_path, run_name, test_root=None):
        root = Path(split_path).resolve()
        train_images = root / "train" / "images"
        val_images = root / "val" / "images"

        for image_dir in (train_images, val_images):
            if not image_dir.is_dir():
                raise FileNotFoundError(
                    f"Image directory not found: {image_dir}"
                )

        self._validate_labels(root / "train")
        self._validate_labels(root / "val")

        test_root = Path(test_root or split_path).resolve()
        test_images = test_root / "test" / "images"
        self.has_test = test_images.is_dir()

        if self.has_test:
            self._validate_labels(test_root / "test")

        data = {
            "path": root.as_posix(),
            "train": "train/images",
            "val": "val/images",
            "names": {
                index: name
                for index, name in enumerate(self.classes)
            },
        }

        if self.has_test:
            data["test"] = test_images.as_posix()

        yaml_path = self.output_path / f"dataset_{run_name}.yaml"

        with yaml_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                data,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

        self.data_yaml = yaml_path
        return yaml_path

    @staticmethod
    def _extract_metrics(results):
        seg = getattr(results, "seg", None)

        if seg is None:
            raise RuntimeError(
                "Validation did not return segmentation metrics."
            )

        return {
            "map": float(seg.map),
            "map50": float(seg.map50),
            "map75": float(seg.map75),
        }

    def _selection_key(self):
        metric = (
            self.selection_metric.lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

        aliases = {
            "map": "map",
            "map5095": "map",
            "map50": "map50",
            "map75": "map75",
        }

        if metric not in aliases:
            raise ValueError(
                f"Selection metric not supported: "
                f"{self.selection_metric}. Use map, map50 or map75."
            )

        return aliases[metric]

    def load_yolo_model(self, weights_name: str) -> YOLO:
        pretrained_dir = self.output_path / "pretrained"
        
        pretrained_dir.mkdir(parents=True, exist_ok=True)

        weights_path = pretrained_dir / Path(weights_name).name

        if not weights_path.exists():
            attempt_download_asset(str(weights_path))

        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Could not download or find YOLO weights: {weights_path}"
            )

        return YOLO(str(weights_path))

    def run_training(self, split_path, run_name, test_root=None):
        data_yaml = self.create_dataset_yaml(
            split_path=split_path,
            run_name=run_name,
            test_root=test_root,
        )
        model = self.load_yolo_model(
            weights_name = self.weights_name
        )
        project_dir = self.output_path / "ultralytics"
        project_dir.mkdir(parents=True, exist_ok=True)

        run_name = Path(self.weights_name).stem

        start = time.perf_counter()

        model.train(
            data=str(data_yaml),
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch_size,
            patience=self.patience,
            workers=self.workers,
            augmentations=get_yolo_augmentation(image_fill=114),
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            close_mosaic=0,
            fliplr=0.0,
            flipud=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,

            seed=self.seed,
            pretrained=True,
            save_period=-1,
            device=self.device,
            project=str(project_dir),
            name=run_name,
            plots=True,
            verbose=False,
            save=True,
        )

        self.training_time = time.perf_counter() - start

        best = Path(model.trainer.best).resolve()

        if not best.is_file():
            raise FileNotFoundError(
                f"Best checkpoint not generated: {best}"
            )

        self.best_weights = best

        evaluator = YOLO(str(best))
        val_results = evaluator.val(
            data=str(data_yaml),
            split="val",
            imgsz=self.imgsz,
            batch=self.batch_size,
            device=self.device,
            plots=False,
            verbose=False,
        )

        self.validation_metrics = self._extract_metrics(val_results)

        score = self.validation_metrics[self._selection_key()]
        logger.info(
            "%s: validation %s = %.4f",
            self.weights_name,
            self._selection_key(),
            score,
        )

        return score, str(best)

    def evaluate(self, split="test"):
        if self.best_weights is None or self.data_yaml is None:
            raise RuntimeError("You must train the model before evaluating it.")

        if split == "test" and not self.has_test:
            raise FileNotFoundError(
                "Test partition not found in the dataset."
            )

        evaluator = YOLO(str(self.best_weights))
        results = evaluator.val(
            data=str(self.data_yaml),
            split=split,
            imgsz=self.imgsz,
            batch=self.batch_size,
            device=self.device,
            plots=False,
            verbose=False,
        )

        return self._extract_metrics(results)
