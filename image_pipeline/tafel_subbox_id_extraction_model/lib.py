"""Utility functions for Detectron2 dataset preparation and processing."""

import json
import os
from typing import Any

import cv2
import numpy as np
from detectron2.structures import (  # type: ignore[import-untyped, import-not-found]
    BoxMode,
)


def get_id_dicts(img_dir: str) -> list[dict[str, Any]]:
    """Load and convert annotation files to Detectron2 dataset format."""
    dataset_dicts = []
    files = [f for f in os.listdir(img_dir) if f.endswith(".json")]

    for idx, json_file in enumerate(files):
        with open(os.path.join(img_dir, json_file)) as f:
            imgs_anns = json.load(f)

        record: dict[str, Any] = {}
        filename = os.path.join(img_dir, imgs_anns["imagePath"])

        # Verify image exists
        if not os.path.exists(filename):
            continue

        img = cv2.imread(filename)
        if img is None:
            continue

        height, width = img.shape[:2]

        record["file_name"] = filename
        record["image_id"] = str(idx)
        record["height"] = height
        record["width"] = width

        objs = []
        for shape in imgs_anns["shapes"]:
            # we annotated the images with the "id" label
            if shape["label"] != "id":
                continue

            points = shape["points"]
            px = [x[0] for x in points]
            py = [x[1] for x in points]

            # convert rectangles to polygons because they only have 2 vertices
            if len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                poly = [x1, y1, x2, y1, x2, y2, x1, y2]
            else:
                poly = [p for point in points for p in point]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
