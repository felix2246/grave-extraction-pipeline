"""Utility functions for Detectron2 dataset preparation and processing."""

import json
import os
from typing import Any

import cv2
import numpy as np
from detectron2.structures import BoxMode  # type: ignore[import-untyped]


def get_box_dicts(img_dirs: list[str]) -> list[dict[str, Any]]:
    """Load and convert annotation files from multiple directories to Detectron2 format."""
    dataset_dicts = []

    # We use a global counter to ensure image_ids are unique across all directories
    global_idx = 0

    for img_dir in img_dirs:
        # Verify directory exists before listing
        if not os.path.isdir(img_dir):
            print(f"Warning: Directory not found: {img_dir}")
            continue

        files = [f for f in os.listdir(img_dir) if f.endswith(".json")]

        for json_file in files:
            json_path = os.path.join(img_dir, json_file)

            with open(json_path) as f:
                imgs_anns = json.load(f)

            record: dict[str, Any] = {}

            # Construct full path based on the CURRENT directory in the loop
            filename = os.path.join(img_dir, imgs_anns["imagePath"])

            # Verify image exists
            if not os.path.exists(filename):
                continue

            # Read image to get dimensions
            img = cv2.imread(filename)
            if img is None:
                continue

            height, width = img.shape[:2]

            record["file_name"] = filename
            record["image_id"] = str(global_idx)  # Use the global counter
            record["height"] = height
            record["width"] = width

            objs = []
            for shape in imgs_anns["shapes"]:
                if shape["label"] != "box":
                    continue

                points = shape["points"]
                px = [x[0] for x in points]
                py = [x[1] for x in points]

                # Convert rectangles (2 points) to polygons (4 points)
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

            global_idx += 1

    return dataset_dicts
