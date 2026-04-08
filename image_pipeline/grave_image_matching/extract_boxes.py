"""Box extraction utilities for post-processing 'Tafel' images with Detectron2."""

import os
from pathlib import Path

import cv2
import numpy as np
from detectron2.config import get_cfg  # type: ignore[import-untyped, import-not-found]
from detectron2.engine import (  # type: ignore[import-untyped, import-not-found]
    DefaultPredictor,
)

from grave_image_matching.logger import logger


def _build_predictor(seg_model_dir: str) -> DefaultPredictor:
    """
    Build a Detectron2 predictor from a given segmentation model directory.

    The directory is expected to contain `config.yaml` and `model_final.pth`.
    """
    seg_dir = Path(seg_model_dir)
    config_path = seg_dir / "config.yaml"
    weights_path = seg_dir / "model_final.pth"

    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    return DefaultPredictor(cfg)


def extract_from_image(
    image_path: str,
    output_base_dir: str,
    seg_model_dir: str = "tafel_segmentation_model/output",
) -> None:
    """Extract detected boxes from an image and save them as individual PNG files.

    Args:
        image_path: Path to the input image file.
        output_base_dir: Base directory where extracted boxes will be saved.
        cfg: Detectron2 configuration object for the model.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    predictor = _build_predictor(seg_model_dir)
    outputs = predictor(img)

    instances = outputs["instances"].to("cpu")
    masks = instances.pred_masks.numpy()
    boxes = instances.pred_boxes.tensor.numpy()

    os.makedirs(output_base_dir, exist_ok=True)

    logger.info("Found boxes", count=len(masks), image=image_path)

    for i, mask in enumerate(masks):
        x1, y1, x2, y2 = boxes[i].astype(int)

        extracted_item = np.full_like(img, 255)
        extracted_item[mask] = img[mask]

        h, w, _ = img.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        final_crop = extracted_item[y1:y2, x1:x2]

        save_path = os.path.join(output_base_dir, f"box_{i}.png")
        cv2.imwrite(save_path, final_crop)
