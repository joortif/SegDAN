"""Convert binary segmentation masks (background/foreground) to YOLO-seg TXT labels.

Input masks must be single-channel binary images encoded by default as:
    0   -> background
    255 -> foreground

One output TXT file is generated per input mask. The output uses YOLO instance-
segmentation polygon syntax, with the foreground represented as class 0.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

from segdan.converters.converter import Converter


class BinaryToYOLOConverter(Converter):

    SUPPORTED_MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        input_data: str,
        output_dir: str,
        background: int = 0,
        foreground: int = 255,
    ):
        super().__init__(input_data, output_dir)

        if not (0 <= background <= 255):
            raise ValueError("background_value must be in the range [0, 255].")
        if not (0 <= foreground <= 255):
            raise ValueError("foreground_value must be in the range [0, 255].")
        if background == foreground:
            raise ValueError("background_value and foreground_value must differ.")

        self.background_value = int(background)
        self.foreground_value = int(foreground)

    def convert(self) -> str:
        mask_dir = Path(self.input_data)
        if not mask_dir.exists() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Binary mask directory not found: {mask_dir}")

        mask_paths = sorted(
            path
            for path in mask_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.SUPPORTED_MASK_EXTENSIONS
        )
        if not mask_paths:
            raise ValueError(
                f"No binary masks found in {mask_dir}. Supported extensions: "
                f"{sorted(self.SUPPORTED_MASK_EXTENSIONS)}"
            )
        seen_stems = set()
        for path in mask_paths:
            stem_key = path.stem.casefold()
            if stem_key in seen_stems:
                raise ValueError(
                    f"Multiple masks share the same filename stem '{path.stem}' "
                    f"in {mask_dir}; output TXT filenames would collide."
                )
            seen_stems.add(stem_key)

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for old_txt in output_path.glob("*.txt"):
            old_txt.unlink()

        with TemporaryDirectory(prefix="segdan_binary_yolo_") as temp_dir:
            normalized_dir = Path(temp_dir)

            for mask_path in mask_paths:
                with Image.open(mask_path) as image:
                    mask = np.asarray(image.convert("L"))

                if mask.ndim != 2:
                    raise ValueError(
                        f"Mask {mask_path.name} is not a single-channel image."
                    )

                values = set(map(int, np.unique(mask)))
                allowed_values = {self.background_value, self.foreground_value}
                unexpected_values = values - allowed_values
                if unexpected_values:
                    raise ValueError(
                        f"Mask {mask_path.name} is not binary with values "
                        f"{self.background_value} (background) and "
                        f"{self.foreground_value} (foreground). "
                        f"Unexpected pixel values: {sorted(unexpected_values)[:20]}"
                    )

                normalized = (mask == self.foreground_value).astype(np.uint8)
                normalized_path = normalized_dir / f"{mask_path.stem}.png"
                Image.fromarray(normalized).save(normalized_path)

            convert_segment_masks_to_yolo_seg(
                masks_dir=str(normalized_dir),
                output_dir=str(output_path),
                classes=1,
            )

        missing_txt = [path.stem for path in mask_paths if not (output_path / f"{path.stem}.txt").exists()]
        if missing_txt:
            raise RuntimeError(
                "Ultralytics did not create TXT labels for these masks: "
                f"{missing_txt[:10]}"
            )

        return str(output_path)