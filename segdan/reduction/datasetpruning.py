from collections import defaultdict
import os
import shutil
import cv2
import numpy as np
from typing import Iterable, Optional, Tuple
import math
from tqdm import tqdm

from imagedatasetanalyzer import ImageDataset

from segdan.converters.converterfactory import ConverterFactory
from segdan.utils.imagelabelutils import ImageLabelUtils
from segdan.utils.confighandler import ConfigHandler
from segdan.utils.constants import LabelFormat

def _normalize_color_tuple(c: Iterable[int]) -> Tuple[int,int,int]:
    c = tuple(int(x) for x in c)
    if len(c) != 3:
        raise ValueError(f"Invalid color (must be 3 component RGB): {c}")
    return c

def png_to_semantic_mask(path: str, colormap: dict = None, background_id: int = 0) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not find {path}")

    if len(img.shape) == 2:
        return img.astype(np.int32)

    if img.shape[2] == 4:
        rgb = img[:, :, :3]

        if np.all(rgb[...,0] == rgb[...,1]) and np.all(rgb[...,1] == rgb[...,2]):
            return rgb[...,0].astype(np.int32)

        img = rgb

    if img.shape[2] == 3:
        h, w, _ = img.shape

        if np.all(img[...,0] == img[...,1]) and np.all(img[...,1] == img[...,2]):
            return img[...,0].astype(np.int32)

        if colormap is None:
            flat = img.reshape(-1, 3)
            colors, counts = np.unique(flat, axis=0, return_counts=True)
            if len(colors) != 2:
                raise ValueError(f"2 colors expected, found {len(colors)}.")
            fg = colors[np.argmin(counts)]
            mask = (img == fg).all(axis=2).astype(np.int32)
            return mask

        cmap = {int(k): _normalize_color_tuple(v) for k, v in colormap.items()}
        color_to_id = {tuple(v): k for k, v in cmap.items()}

        mask = np.full((h, w), background_id, dtype=np.int32)

        for color, cls_id in color_to_id.items():
            match = (img == np.array(color)).all(axis=2)
            mask[match] = cls_id

        return mask

    raise ValueError(f"Invalid PNG, shape={img.shape}")

def extract_instances_from_semantic_mask(sem_mask):
    instances = []
    classes = np.unique(sem_mask)
    for cls in classes:
        if cls == 0:
            continue
        binmask = (sem_mask == cls).astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(binmask, connectivity=8)
        for lab in range(1, num_labels):
            comp_mask = (labels == lab).astype(np.uint8)
            instances.append((int(cls), comp_mask))
    return instances

def contour_perimeter_and_area(binary_mask, name):
    img = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]
    
    perimeter = float(cv2.arcLength(cnt, True))
    area = int(cv2.contourArea(cnt))  

    return perimeter, area

def scs(perimeter, area):
    if perimeter == 0 or area == 0:
        return 0.0
    return perimeter / area

def si_scs(perimeter, area):
    if perimeter == 0 or area == 0:
        return 0.0
    return perimeter / (2.0 * math.sqrt(math.pi * area))

def cb_scs(masks):

    all_instances = []  
    per_image_instances = {}

    pbar = tqdm(total=len(masks), desc="Selecting subset using TFDP (Training-Free Dataset Pruning)")

    for i, mask in masks.items():
        per_image_instances[i] = []
        instances = extract_instances_from_semantic_mask(mask)
        for cls, inst_mask in instances:
            perim, area = contour_perimeter_and_area(inst_mask, i)
            if perim == 0 or area == 0:
                continue
            scs_mask = scs(perim, area)
            si_mask = si_scs(perim, area)
            all_instances.append((i, cls, si_mask))
            per_image_instances[i].append({'class': cls, 'area': area, 'perimeter': perim, 'scs': scs_mask, 'si_scs': si_mask, 'mask': inst_mask})
        
        pbar.update(1)

    pbar.close()
    class_totals = defaultdict(float)
    for (img_idx, cls, si) in all_instances:
        class_totals[cls] += float(si)

    image_cb_scores = {}
    for mask_name, cls, si_val in all_instances:
        denom = class_totals.get(cls)
        norm_score = si_val / denom
        image_cb_scores[mask_name] = image_cb_scores.get(mask_name, 0.0) + norm_score

    return image_cb_scores, per_image_instances, class_totals

def prune_dataset(general_config: dict, dataset: ImageDataset, retention_percentage: float, label_dir: str, output_dir: str):

    label_extension = ImageLabelUtils.check_label_extensions(label_dir, verbose=False)

    if label_extension.lower() not in ConfigHandler.VALID_IMAGE_EXTENSIONS:
        output_dir_transformations = os.path.join(output_dir, "transformations", label_extension.lower())

        args = {
            "input_data": label_dir,
            "output_dir": output_dir_transformations,
            "background": general_config.get('background', None),
            "threshold": general_config.get('threshold', None),
            "color_dict": general_config.get('color_dict', None),
            "img_dir": dataset.img_dir
        }        
        
        converter = ConverterFactory().get_converter(label_extension.lower(), LabelFormat.MASK.value, args)
        converter.convert()

        label_dir = output_dir_transformations
    
    masks = {}
    for label in os.listdir(label_dir):
        
        label_name = os.path.splitext(label)[0]
        ext = os.path.splitext(label)[1].lower()
        if ext in ConfigHandler.VALID_IMAGE_EXTENSIONS:   
            masks[label_name] = png_to_semantic_mask(os.path.join(label_dir, label))

    image_cb_scores, _, _ = cb_scs(masks)
    scores_sorted = sorted(image_cb_scores.items(), key=lambda kv: (kv[1], kv[0]))

    imgs_to_select = int(np.floor(retention_percentage * len(dataset.image_files)))
    top_images = scores_sorted[:imgs_to_select]

    filenames = [f"{name}.png" for name, _ in top_images]
    reduced_ds = ImageDataset(output_dir, filenames)

    for filename in filenames:
        src_path = os.path.join(dataset.img_dir, filename)
        dst_path = os.path.join(reduced_ds.img_dir, filename)
        shutil.copy(src_path, dst_path)
    
    return reduced_ds 