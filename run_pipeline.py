from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import cv2
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import yaml

from imagedatasetanalyzer import ImageDataset

from segdan.analysis.analysis import analyze_data
from segdan.clustering.clustering import cluster_images, get_embeddings
from segdan.converters.converterfactory import ConverterFactory
from segdan.datasets.traindataset import TrainingDataset
from segdan.reduction.reduction import reduce_dataset
from segdan.training.training import model_training
from segdan.utils.constants import LabelFormat, ReductionMethods, SegmentationType
from segdan.utils.imagelabelutils import ImageLabelUtils
from segdan.utils.confighandler import ConfigHandler


IMAGE_EXTENSIONS = tuple(ConfigHandler.VALID_IMAGE_EXTENSIONS)
_MASK_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}


def create_logger(output_path: str | None = None, verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger("SegDAN.Pipeline")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(output_path) / "segdan_pipeline.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            config = yaml.safe_load(f)
        elif suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError("Config file must have a .yaml, .yml, or .json extension.")

    if not isinstance(config, dict) or "general_data" not in config:
        raise ValueError(
            "This runner expects a full config exported by SegDAN Wizard, with a top-level "
            "'general_data' section. It does not consume the separate ConfigHandler config.yaml schema."
        )
    return config


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _class_mapping(general_data: dict[str, Any]) -> dict[int, str]:
    mapping = general_data.get("class_mapping") or {}
    result: dict[int, str] = {}
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            try:
                result[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
    return result


def _read_yolo_class_ids(label_dir: str) -> set[int]:
    """Read zero-based class IDs from YOLO TXT files (case-insensitive extension)."""
    root = Path(label_dir)
    txt_files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    class_ids: set[int] = set()
    for txt_file in sorted(txt_files):
        with txt_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                values = line.split()
                try:
                    class_id = int(values[0])
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"Invalid YOLO label at {txt_file}:{line_no}: {line!r}") from exc
                if class_id < 0:
                    raise ValueError(f"Negative class ID at {txt_file}:{line_no}.")
                class_ids.add(class_id)
    return class_ids


def infer_class_names(general_data: dict[str, Any], label_format: str, label_dir: str) -> list[str]:
    mapping = _class_mapping(general_data)
    background = general_data.get("background")
    if background is not None:
        background = _safe_int(background, -999999)

    if mapping:
        if label_format == LabelFormat.TXT.value:
            class_ids = sorted(mapping)
        else:
            class_ids = sorted(cid for cid in mapping if cid != background)
        if class_ids and class_ids == list(range(class_ids[-1] + 1)):
            return [mapping[cid] for cid in class_ids]
        max_id = max(class_ids, default=-1)
        names = [f"class_{i}" for i in range(max_id + 1)]
        for cid in class_ids:
            names[cid] = mapping[cid]
        return names

    if label_format == LabelFormat.TXT.value:
        ids = _read_yolo_class_ids(label_dir)
        if not ids:
            raise ValueError(
                f"No class IDs found in .txt labels under {label_dir}. "
                "Check that the labels contain YOLO polygon annotations."
            )

        return [f"class_{i}" for i in range(max(ids) + 1)]

    if label_format in {LabelFormat.MASK.value, LabelFormat.BINARY.value, LabelFormat.COLOR.value}:
        mask_classes = set()
        label_path = Path(label_dir)
        for file_path in sorted(p for p in label_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
            
            mask = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            if mask.ndim == 3:
                continue
            mask_classes.update(int(v) for v in np.unique(mask))
        if background is not None:
            mask_classes.discard(background)
        ids = sorted(mask_classes)
        if not ids:
            raise ValueError("Could not infer classes from masks; provide a class_mapping in the Wizard config.")
        return [f"class_{i}" for i in ids]

    raise ValueError(
        f"Cannot infer class names for label format '{label_format}'. "
        "Provide class_mapping in the Wizard config."
    )


def _paired_yolo_files(image_dir: str, label_dir: str) -> list[tuple[Path, Path | None]]:
    image_root = Path(image_dir)
    label_root = Path(label_dir)
    if not image_root.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {image_root}")
    if not label_root.is_dir():
        raise NotADirectoryError(f"Label directory does not exist: {label_root}")

    images = sorted(p for p in image_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    labels_by_stem = {p.stem: p for p in label_root.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}
    if not images:
        raise ValueError(f"No images found in {image_root}")
    if not labels_by_stem:
        raise ValueError(f"No YOLO .txt labels found in {label_root}")

    image_stems = {p.stem for p in images}
    orphan_labels = sorted(set(labels_by_stem) - image_stems)
    if orphan_labels:
        raise ValueError(f"YOLO labels without matching images (first 10): {orphan_labels[:10]}")

    return [(image, labels_by_stem.get(image.stem)) for image in images]


def split_yolo_dataset(
    image_dir: str,
    label_dir: str,
    split_output_path: str,
    split_data: dict[str, Any],
    class_names: list[str],
    logger: logging.Logger,
) -> str:
    if not _as_bool(split_data.get("split_method", True)):
        raise NotImplementedError("The YOLO smoke runner currently supports hold-out splitting, not cross-validation.")
    if _as_bool(split_data.get("stratification", False)):
        raise NotImplementedError(
            "For direct YOLO .txt input, set split_data.stratification=false for this smoke test. "
            "YOLO polygon-aware stratification is not implemented in this runner."
        )

    split_percentages = split_data.get("hold_out", {})
    train_p = float(split_percentages.get("train", 0.7))
    val_p = float(split_percentages.get("valid", 0.2))
    test_p = float(split_percentages.get("test", 0.1))
    if min(train_p, val_p, test_p) < 0 or not np.isclose(train_p + val_p + test_p, 1.0):
        raise ValueError("Hold-out train/valid/test fractions must be non-negative and sum to 1.0.")

    pairs = _paired_yolo_files(image_dir, label_dir)
    seed = _safe_int(split_data.get("stratification_random_seed", 123), 123)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs)).tolist()
    shuffled = [pairs[i] for i in order]

    n = len(shuffled)
    n_train = int(n * train_p)
    n_val = int(n * val_p)
    n_test = n - n_train - n_val
    if n_train == 0 or (val_p > 0 and n_val == 0) or (test_p > 0 and n_test == 0):
        logger.warning(
            "Small dataset (%d images) produced an empty requested split: train=%d, val=%d, test=%d. "
            "Consider using fewer splits for this smoke test.", n, n_train, n_val, n_test
        )

    subsets = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }
    if n_train == 0 or n_val == 0:
        raise ValueError(
            f"The dataset split produced an empty train or validation set (train={n_train}, val={n_val}). "
            "Use more images or adjust the train/validation percentages."
        )

    out_root = Path(split_output_path).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for subset_name, subset_pairs in subsets.items():
        subset_root = out_root / subset_name
        if subset_root.exists():
            shutil.rmtree(subset_root)
        images_out = subset_root / "images"
        labels_out = subset_root / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        for image_file, label_file in subset_pairs:
            shutil.copy2(image_file, images_out / image_file.name)
            destination_label = labels_out / f"{image_file.stem}.txt"
            if label_file is None:
                destination_label.touch()  # explicit empty label for a negative image
            else:
                shutil.copy2(label_file, destination_label)
        logger.info("YOLO split %-5s: %d images", subset_name, len(subset_pairs))

    data_yaml = {
        "path": str(out_root),
        "train": "train/images",
        "val": "val/images",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    if subsets["test"]:
        data_yaml["test"] = "test/images"

    data_yaml_path = out_root / "data.yaml"
    with data_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    logger.info("Wrote Ultralytics dataset configuration: %s", data_yaml_path)
    return str(data_yaml_path)



def _reset_directory(directory: str | Path) -> Path:
    path = Path(directory)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def _iter_mask_files(label_dir: str | Path) -> list[Path]:
    root = Path(label_dir)
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in _MASK_EXTENSIONS
    )


def _resolve_json_file(label_path: str | Path) -> Path:
    path = Path(label_path)
    if path.is_file() and path.suffix.lower() == ".json":
        return path
    if path.is_dir():
        matches = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".json")
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"No COCO JSON annotation file found in {path}")
        raise ValueError(
            f"More than one JSON file was found in {path}: {[p.name for p in matches]}. "
            "Set general_data.label_path to the specific annotation JSON file."
        )
    raise FileNotFoundError(f"JSON annotation file or directory not found: {path}")


def _detect_label_format(label_path: str | Path, configured_format: Any = None, class_mapping: Any = None) -> str:
    """Resolve the real input format, correcting stale config values when extensions disagree."""
    configured = str(configured_format or "auto").strip().lower()
    supported = {fmt.value for fmt in LabelFormat}
    path = Path(label_path)

    # A standalone annotation file has an unambiguous format from its suffix.
    if path.is_file():
        extension = path.suffix.lower()
        if extension == ".json":
            return LabelFormat.JSON.value
        if extension == ".txt":
            return LabelFormat.TXT.value
        if extension in _MASK_EXTENSIONS:
            return configured if configured in {LabelFormat.MASK.value, LabelFormat.BINARY.value, LabelFormat.COLOR.value} else LabelFormat.MASK.value

    if not path.is_dir():
        raise FileNotFoundError(f"Label path does not exist or is not a supported file/directory: {path}")

    children = [p for p in path.iterdir() if p.is_file()]
    txt_files = [p for p in children if p.suffix.lower() == ".txt"]
    json_files = [p for p in children if p.suffix.lower() == ".json"]
    image_labels = [p for p in children if p.suffix.lower() in _MASK_EXTENSIONS]

    # Configured format does not match the files; use the actual contents.
    if txt_files:
        return LabelFormat.TXT.value
    if json_files:
        if len(json_files) > 1:
            raise ValueError(
                f"Multiple JSON annotation files found in {path}: {[p.name for p in json_files]}. "
                "Set general_data.label_path to the specific annotation JSON file."
            )
        return LabelFormat.JSON.value
    if image_labels:
        
        sample = cv2.imread(str(image_labels[0]), cv2.IMREAD_UNCHANGED)
        if sample is None:
            raise ValueError(f"Could not read label file: {image_labels[0]}")
        if sample.ndim == 3:
            channels = sample[..., :3]
            if not (np.array_equal(channels[..., 0], channels[..., 1]) and np.array_equal(channels[..., 0], channels[..., 2])):
                return LabelFormat.COLOR.value

        class_mapping = class_mapping or {}
        mapped_values = {int(k) for k in class_mapping}
        pixel_values = set(np.unique(sample).astype(int))

        if len(mapped_values) == 2 and pixel_values.issubset(mapped_values):
            return LabelFormat.BINARY.value

        return LabelFormat.MASK.value

    raise ValueError(
        f"Cannot infer annotation format from {path}. Set general_data.label_format to one of "
        f"{sorted(supported)} and ensure the directory contains matching label files."
    )


def _canonicalize_mask_directory(source_dir: str | Path, output_dir: str | Path) -> str:
    """Copy masks to a clean directory using lowercase .png names and one channel."""
    

    source = Path(source_dir)
    if not source.is_dir():
        raise NotADirectoryError(f"Mask directory does not exist: {source}")
    mask_files = _iter_mask_files(source)
    if not mask_files:
        raise ValueError(f"No image masks found in {source} (extensions are case-insensitive).")

    output = _reset_directory(output_dir)
    seen_stems: set[str] = set()
    for file_path in mask_files:
        if file_path.stem in seen_stems:
            raise ValueError(f"Multiple mask files share the same base name: {file_path.stem!r}")
        seen_stems.add(file_path.stem)
        mask = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Could not read mask: {file_path}")
        if mask.ndim == 3:
            channels = mask[..., :3]
            if channels.ndim == 3 and np.array_equal(channels[..., 0], channels[..., 1]) and np.array_equal(channels[..., 0], channels[..., 2]):
                mask = channels[..., 0]
            else:
                raise ValueError(
                    f"Mask {file_path.name} is a colour image. Set label_format='color' and provide "
                    "a color_dict, or convert it to a class-ID mask first."
                )
        destination = output / f"{file_path.stem}.png"
        if not cv2.imwrite(str(destination), mask):
            raise OSError(f"Could not write canonical PNG mask: {destination}")

    return str(output)


def _validate_image_mask_pairs(image_dir: str | Path, mask_dir: str | Path) -> None:
    image_stems = {
        p.stem for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }
    mask_stems = {p.stem for p in _iter_mask_files(mask_dir)}
    missing_masks = sorted(image_stems - mask_stems)
    orphan_masks = sorted(mask_stems - image_stems)
    if missing_masks or orphan_masks:
        parts = []
        if missing_masks:
            parts.append(f"images without masks (first 10): {missing_masks[:10]}")
        if orphan_masks:
            parts.append(f"masks without images (first 10): {orphan_masks[:10]}")
        raise ValueError("Image/mask mismatch after label conversion: " + "; ".join(parts))


def _coerce_color_dict(raw: Any) -> dict[tuple[int, int, int], int] | None:
    """Parse a YAML/JSON colour map. Colour tuples follow OpenCV's BGR order."""
    if not isinstance(raw, dict):
        return None
    result: dict[tuple[int, int, int], int] = {}
    for key, value in raw.items():
        parsed: Any = key
        if isinstance(key, str):
            try:
                parsed = ast.literal_eval(key)
            except (ValueError, SyntaxError):
                stripped = key.strip().strip("()[]")
                try:
                    parsed = [int(part.strip()) for part in stripped.split(",")]
                except ValueError:
                    continue
        if isinstance(parsed, (tuple, list)) and len(parsed) >= 3:
            result[tuple(int(v) for v in parsed[:3])] = int(value)
    return result or None


def _infer_binary_threshold(label_dir: str | Path) -> int:
    """Choose a usable threshold for common 0/1 and 0/255 binary masks."""
    
    values: set[int] = set()
    for path in _iter_mask_files(label_dir):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            values.update(int(v) for v in np.unique(mask))
    nonzero = sorted(v for v in values if v > 0)
    if not nonzero:
        raise ValueError(f"No foreground pixels detected in binary masks under {label_dir}.")
    return 1 if max(nonzero) <= 1 else nonzero[0]



def prepare_mask_labels(
    image_dir: str,
    label_path: str,
    source_format: str,
    output_path: str,
    general_data: dict[str, Any],
    converter_factory: ConverterFactory,
    logger: logging.Logger,
) -> str:
    """Convert any supported annotation source to a canonical PNG mask directory."""
    source_format = _detect_label_format(label_path, source_format, general_data.get("class_mapping"))
    transformations_dir = Path(output_path) / "transformations"
    raw_dir = transformations_dir / f"_raw_masks_from_{source_format}"
    canonical_dir = transformations_dir / "canonical_masks"
    background = general_data.get("background")
    if background is None:
        background = 0
        general_data["background"] = 0

    if source_format == LabelFormat.TXT.value:
        
        yolo_class_ids = _read_yolo_class_ids(label_path)
        if 255 in yolo_class_ids:
            raise ValueError(
                "YOLO class ID 255 cannot be represented alongside a distinct "
                "background in an 8-bit class-ID mask."
            )
        if _safe_int(background, 0) in yolo_class_ids:
            background = 255
            general_data["background"] = background
            logger.warning(
                "Configured background value overlaps a YOLO class ID; using "
                "background=255 for TXT-to-MASK conversion. YOLO class IDs "
                "remain zero-based."
            )

    if source_format == LabelFormat.MASK.value:
        source_masks = _iter_mask_files(label_path)
        if not source_masks:
            raise ValueError(f"No image masks found in {label_path} (extensions are case-insensitive).")
        already_canonical = all(p.suffix == ".png" for p in source_masks)
        if already_canonical:
            
            for file_path in source_masks:
                mask = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    raise ValueError(f"Could not read mask: {file_path}")
                if mask.ndim != 2:
                    already_canonical = False
                    break
        mask_dir = str(Path(label_path)) if already_canonical else _canonicalize_mask_directory(label_path, canonical_dir)
    else:

        _reset_directory(raw_dir)
        if source_format == LabelFormat.TXT.value:
            context = {
                "input_data": label_path,
                "output_dir": str(raw_dir),
                "img_dir": image_dir,
                "background": background,
                "depth_model": general_data.get(
                    "depth_model", "Intel/dpt-swinv2-tiny-256"
                ),
            }
        elif source_format == LabelFormat.JSON.value:
            input_data = str(_resolve_json_file(label_path))
            context = {
                "input_data": input_data,
                "output_dir": str(raw_dir),
                "img_dir": image_dir,
                "background": background,
                "depth_model": general_data.get("depth_model", "Intel/dpt-swinv2-tiny-256"),
            }
        elif source_format == LabelFormat.BINARY.value:
            threshold = general_data.get("binary_threshold", general_data.get("threshold"))
            if threshold is None:
                threshold = _infer_binary_threshold(label_path)
            context = {
                "input_data": label_path,
                "output_dir": str(raw_dir),
                "threshold": int(threshold),
            }
            # BinaryToMultilabelConverter emits background=0 and foreground=1.
            general_data["background"] = 0
        elif source_format == LabelFormat.COLOR.value:
            color_dict = _coerce_color_dict(general_data.get("color_dict"))
            if not color_dict:
                raise ValueError(
                    "For label_format='color', automatic routing is supported but a color_dict is required "
                    "to map each BGR colour to a class ID. Add general_data.color_dict to the config."
                )
            context = {
                "input_data": label_path,
                "output_dir": str(raw_dir),
                "color_dict": color_dict,
            }
        else:
            raise ValueError(f"No conversion route from {source_format!r} to PNG masks.")

        converter = converter_factory.get_converter(
            source_format, LabelFormat.MASK.value, context
        )
        converter.convert()
        mask_dir = _canonicalize_mask_directory(raw_dir, canonical_dir)

    _validate_image_mask_pairs(image_dir, mask_dir)
    logger.info("Labels ready in canonical PNG mask format: %s", mask_dir)
    return mask_dir


def _infer_mask_class_values(label_dir: str, background: Any) -> list[int]:
    

    bg = 0 if background is None else _safe_int(background, 0)
    values: set[int] = set()
    for file_path in _iter_mask_files(label_dir):
        mask = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Could not read mask: {file_path}")
        if mask.ndim != 2:
            raise ValueError(
                f"Mask {file_path} must contain one channel of class IDs; shape received: {mask.shape}."
            )
        values.update(int(value) for value in np.unique(mask))
    values.discard(bg)
    if not values:
        raise ValueError(f"No foreground class values found in masks under {label_dir}; background={bg}.")
    return sorted(values)


def convert_masks_to_yolo_labels(
    label_dir: str,
    output_dir: str,
    general_data: dict[str, Any],
    converter_factory: ConverterFactory,
) -> tuple[str, list[str]]:
    

    background = general_data.get("background", 0)
    if background is None:
        background = 0
    mapping = _class_mapping(general_data)
    class_values = _infer_mask_class_values(label_dir, background)
    if len(class_values) > 255:
        raise ValueError("YOLO mask conversion supports at most 255 foreground classes with 8-bit masks.")

    if mapping and all(value in mapping for value in class_values):
        class_names = [mapping[value] for value in class_values]
    else:
        class_names = [mapping.get(index, f"class_{index}") for index, _ in enumerate(class_values)]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for old_txt in output.iterdir():
        if old_txt.is_file() and old_txt.suffix.lower() == ".txt":
            old_txt.unlink()

    with TemporaryDirectory(prefix="segdan_yolo_mask_normalization_") as temporary_dir:
        normalized_dir = Path(temporary_dir) / "masks"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        source_masks = _iter_mask_files(label_dir)
        if not source_masks:
            raise ValueError(f"No masks to convert in {label_dir}")
        for mask_path in source_masks:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise ValueError(f"Could not read mask: {mask_path}")
            if mask.ndim != 2:
                raise ValueError(
                    f"Mask {mask_path.name} is not a one-channel class-ID mask. "
                    "Colour masks must first be converted with label_format='color'."
                )
            normalized = np.zeros(mask.shape, dtype=np.uint8)  
            for yolo_index, source_value in enumerate(class_values):
                normalized[mask == source_value] = yolo_index + 1
            destination = normalized_dir / f"{mask_path.stem}.png"
            if not cv2.imwrite(str(destination), normalized):
                raise OSError(f"Could not write normalized mask: {destination}")

        converter = converter_factory.get_converter(
            LabelFormat.MASK.value,
            LabelFormat.TXT.value,
            {
                "input_data": str(normalized_dir),
                "output_dir": str(output),
                "num_classes": len(class_values),
            },
        )
        converter.convert()

    for mask_path in _iter_mask_files(label_dir):
        (output / f"{mask_path.stem}.txt").touch(exist_ok=True)

    return str(output), class_names


def _is_yolo_instance_model(model_data: dict[str, Any]) -> bool:
    segmentation = str(model_data.get("segmentation", "")).strip().lower()
    models = model_data.get("models") or []
    if isinstance(models, dict):
        models = [models]
    return segmentation == SegmentationType.INSTANCE.value and any(
        "yolo" in str(model.get("model_name", "")).strip().lower()
        for model in models if isinstance(model, dict)
    )

def _resolve_binary_class_mapping(
    general_data: dict[str, Any],
) -> tuple[int, int, str]:
    """Return (background pixel value, foreground pixel value, foreground name)."""
    mapping = _class_mapping(general_data)

    if len(mapping) != 2:
        raise ValueError(
            "BinaryToYOLOConverter requires exactly two entries in "
            f"'class_mapping': one background and one foreground. Got: {mapping}"
        )

    background_names = {"background", "bg", "fondo"}

    backgrounds = [
        (pixel_value, name)
        for pixel_value, name in mapping.items()
        if name.strip().casefold() in background_names
    ]

    if len(backgrounds) == 1:
        background_value = backgrounds[0][0]
    elif not backgrounds:
        configured_background = general_data.get("background")

        try:
            background_value = int(configured_background)
        except (TypeError, ValueError):
            raise ValueError(
                "Cannot identify the background class. Name one entry "
                "of class_mapping 'background' or configure its pixel value "
                "in general_data['background']."
            ) from None

        if background_value not in mapping:
            raise ValueError(
                f"Configured background value {background_value} is not "
                f"present in class_mapping: {mapping}"
            )
    else:
        raise ValueError(
            f"Multiple background classes found in class_mapping: {backgrounds}"
        )

    foregrounds = [
        (pixel_value, name)
        for pixel_value, name in mapping.items()
        if pixel_value != background_value
    ]

    if len(foregrounds) != 1:
        raise ValueError(
            f"Expected exactly one foreground class, got: {foregrounds}"
        )

    foreground_value, foreground_name = foregrounds[0]

    return background_value, foreground_value, foreground_name


def run_pipeline(config_dict: dict[str, Any]) -> None:
    if not isinstance(config_dict, dict) or "general_data" not in config_dict:
        raise ValueError("Expected a Wizard-exported config with 'general_data'.")

    general_data = config_dict["general_data"]
    output_path = str(general_data["output_path"])
    Path(output_path).mkdir(parents=True, exist_ok=True)
    logger = create_logger(output_path, _as_bool(general_data.get("verbose", True)))
    converter_factory = ConverterFactory()

    image_path = str(general_data["image_path"])
    label_path = str(general_data["label_path"])
    background = general_data.get("background")
    verbose = _as_bool(general_data.get("verbose", True))
    class_mapping = _class_mapping(general_data)

    if not Path(image_path).is_dir():
        raise NotADirectoryError(f"Image path does not exist: {image_path}")
    if not Path(label_path).exists():
        raise FileNotFoundError(f"Label path does not exist: {label_path}")

    label_format = _detect_label_format(label_path, general_data.get("label_format", "auto"), class_mapping)
    model_data = config_dict.get("model_data", {})
    split_data = config_dict.get("split_data", {})
    yolo_instance = _is_yolo_instance_model(model_data)

    logger.info("Starting SegDAN pipeline")
    logger.info("Task: %s; labels: %s", model_data.get("segmentation"), label_format)
    logger.info("Images: %s", image_path)
    logger.info("Labels: %s", label_path)

    image_dataset = ImageDataset(image_path)
    embeddings_dict = None
    clustering_results = None
    clustering_metric = None
    height_mode = width_mode = None
    classes: list[str] | None = list(class_mapping.values()) if class_mapping else None

    # Stage 1: dataset analysis (optional).
    if _as_bool(config_dict.get("analyze", False)):
        logger.info("Dataset analysis enabled")
        height_mode, width_mode = analyze_data(
            general_data, converter_factory, output_path, class_mapping, verbose
        )
        metrics_csv = Path(output_path) / "analysis" / "metrics.csv"
        if metrics_csv.exists():
            classes = ImageLabelUtils.get_classes_from_csv(str(metrics_csv), background)

    # Stage 2: clustering / embeddings (optional).
    if _as_bool(config_dict.get("cluster_images", False)):
        clustering_data = config_dict.get("clustering_data") or {}
        logger.info("Extracting embeddings")
        embeddings = get_embeddings(clustering_data, image_dataset, verbose)
        embeddings_dict = {name: embedding for name, embedding in zip(image_dataset.image_files, embeddings)}
        clustering_results = cluster_images(
            clustering_data, image_dataset, embeddings, output_path, verbose, logger
        )
        clustering_metric = clustering_data.get("clustering_metric")

    # Stage 3: dataset reduction (optional).
    if _as_bool(config_dict.get("reduce_images", False)):
        reduction_data = config_dict.get("reduction_data") or {}
        reduction_type = reduction_data.get("reduction_type")
        if ReductionMethods.is_embedding(str(reduction_type)) and embeddings_dict is None:
            clustering_data = config_dict.get("clustering_data") or {}
            embeddings = get_embeddings(clustering_data, image_dataset, verbose)
            embeddings_dict = {name: embedding for name, embedding in zip(image_dataset.image_files, embeddings)}
            clustering_metric = clustering_data.get("clustering_metric")

        reduction_path = reduce_dataset(
            config=reduction_data,
            general_config=general_data,
            clustering_results=clustering_results,
            evaluation_metric=clustering_metric,
            dataset=image_dataset,
            label_path=label_path,
            embeddings_dict=embeddings_dict,
            output_path=output_path,
            verbose=verbose,
        )
        if _as_bool(reduction_data.get("use_reduced", False)):
            image_path = os.path.join(reduction_path, "images")
            label_path = os.path.join(reduction_path, "labels")
            # The reduced output may have a different label representation than the source.
            label_format = _detect_label_format(label_path, "auto", class_mapping)
            logger.info("Using reduced dataset: %s", reduction_path)

    # Determine dimensions for legacy training API.
    if height_mode is None or width_mode is None:
        height_mode, width_mode = image_dataset.image_sizes()

    split_output_path = os.path.join(output_path, "split")
    model_output_path = os.path.join(output_path, "models")
    Path(split_output_path).mkdir(parents=True, exist_ok=True)
    Path(model_output_path).mkdir(parents=True, exist_ok=True)

    # Stage 4: adapt labels to the format required by the selected model, then split.
    # Canonical intermediate representation for mask-based models is one-channel PNG.
    # YOLO segmentation consumes polygon TXT; mask -> TXT goes through the same conversion
    # registry after class IDs/background are normalized.
    source_label_format = label_format
    if yolo_instance:
        if source_label_format == LabelFormat.TXT.value:
            # TXT is already YOLO polygon format; preserve instance polygons without a lossy round-trip.
            if not Path(label_path).is_dir():
                raise ValueError("YOLO TXT training expects label_path to be a directory of per-image .txt files.")
            yolo_label_dir = label_path
            classes = infer_class_names(general_data, LabelFormat.TXT.value, yolo_label_dir)
        elif source_label_format == LabelFormat.MASK.value:

            yolo_label_dir = os.path.join(
                output_path, "transformations", "yolo_labels"
            )
            yolo_label_dir, classes = convert_masks_to_yolo_labels(
                label_dir=label_path,
                output_dir=yolo_label_dir,
                general_data=general_data,
                converter_factory=converter_factory,
            )

        elif converter_factory.supports_conversion(
            source_label_format,
            LabelFormat.TXT.value,
        ):
            yolo_label_dir = os.path.join(
                output_path, "transformations", "yolo_labels"
            )

            context = {
                "input_data": str(label_path),
                "output_dir": str(yolo_label_dir),
                "foreground": int(
                    general_data.get("foreground", 255)
                ),
                "img_dir": image_path,
                "background": general_data.get("background", 0),
                "depth_model": general_data.get(
                    "depth_model", "Intel/dpt-swinv2-tiny-256"
                ),
                **(general_data.get("conversion_params") or {}),
            }

            if source_label_format == LabelFormat.BINARY.value:
                (background_value, foreground_value, foreground_name) = _resolve_binary_class_mapping(general_data)

                context["background"] = background_value
                context["foreground"] = foreground_value

            converter = converter_factory.get_converter(
                source_label_format,
                LabelFormat.TXT.value,
                context,
            )
            converter.convert()

            if source_label_format == LabelFormat.BINARY.value:
                classes = [foreground_name]
            else:
                classes = infer_class_names(
                    general_data,
                    source_label_format,
                    str(label_path),
                )

        else:
            # Si no hay conversión directa, convertir primero a MASK y después
            # utilizar el conversor MASK -> TXT que ya existe.
            yolo_source_mask_dir = prepare_mask_labels(
                image_dir=image_path,
                label_path=label_path,
                source_format=source_label_format,
                output_path=output_path,
                general_data=general_data,
                converter_factory=converter_factory,
                logger=logger,
            )

            yolo_label_dir = os.path.join(
                output_path, "transformations", "yolo_labels"
            )
            yolo_label_dir, classes = convert_masks_to_yolo_labels(
                label_dir=yolo_source_mask_dir,
                output_dir=yolo_label_dir,
                general_data=general_data,
                converter_factory=converter_factory,
            )

        data_yaml_path = split_yolo_dataset(
            image_dir=image_path,
            label_dir=yolo_label_dir,
            split_output_path=split_output_path,
            split_data=split_data,
            class_names=classes,
            logger=logger,
        )
        general_data["yolo_data_yaml"] = data_yaml_path
        model_data["data_yaml"] = data_yaml_path
    else:
        # For U-Net/SMP and Hugging Face mask-based models, convert any registered source
        # format to canonical PNG masks automatically before stratification/splitting.
        if source_label_format == LabelFormat.TXT.value:
            # TXT class IDs are zero-based; retain their names before mapping them to pixel values 1..N.
            classes = infer_class_names(general_data, LabelFormat.TXT.value, label_path)

        canonical_mask_dir = prepare_mask_labels(
            image_dir=image_path,
            label_path=label_path,
            source_format=source_label_format,
            output_path=output_path,
            general_data=general_data,
            converter_factory=converter_factory,
            logger=logger,
        )
        if classes is None:
            classes = infer_class_names(general_data, LabelFormat.MASK.value, canonical_mask_dir)

        dataset = TrainingDataset(
            image_path,
            canonical_mask_dir,
            split_output_path,
            LabelFormat.MASK.value,
            canonical_mask_dir,
        )
        dataset.split(general_data, split_data)

    # Stage 5: model training.
    logger.info("Starting model training")
    before_weights = set(Path(model_output_path).rglob("*.pt")) if yolo_instance else set()
    model_training(
        model_data=model_data,
        general_data=general_data,
        split_path=split_output_path,
        model_output_path=model_output_path,
        hold_out=_as_bool(split_data.get("split_method", True)),
        classes=np.asarray(classes) if classes is not None else None,
        mode_height=int(height_mode),
        mode_width=int(width_mode),
    )

    if yolo_instance:
        after_weights = set(Path(model_output_path).rglob("*.pt"))
        new_weights = after_weights - before_weights
        if not new_weights and not after_weights:
            raise RuntimeError(
                "No YOLO .pt checkpoint was created. The runner reached model_training(), but the "
                "instance/YOLO branch may not be connected in your local segdan/training/training.py, "
                "or your adapter may be saving checkpoints outside model_output_path. Check the training logs."
            )
        logger.info("YOLO checkpoint(s): %s", [str(p) for p in sorted(after_weights)])

    logger.info("SegDAN pipeline finished. Results: %s", output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute a SegDAN Wizard configuration end-to-end.")
    parser.add_argument("--config", help="Path to a YAML/JSON configuration exported by the SegDAN Wizard.")
    parser.add_argument("--wizard", action="store_true", help="Open the Wizard and run the generated configuration.")
    args = parser.parse_args()

    if not args.config and not args.wizard:
        parser.error("Provide --config path/to/config.yaml or use --wizard.")

    if args.wizard:
        from segdan.forms.wizard import Wizard
        app = Wizard()
        app.mainloop()
        config_dict = app.final_dict
        try:
            app.destroy()
        except Exception:
            pass
        if not config_dict:
            print("No configuration was produced; exiting.")
            return 1
    else:
        config_dict = load_config(args.config)

    try:
        run_pipeline(config_dict)
    except Exception:
        logging.getLogger("SegDAN.Pipeline").exception("Pipeline failed")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
