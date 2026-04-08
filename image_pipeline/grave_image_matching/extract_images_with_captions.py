"""
Extracts images and their associated captions from a PDF and saves them to disk.
Additionally, generates a JSON file that maps each extracted image to its detected caption for
traceability and downstream use. For images identified as 'Tafel' plates, the script also applies
a trained segmentation model to split these images into subimages, saving each segment separately
for further processing.
"""

import json
import os
import re
from typing import Any, Optional

import fitz  # type: ignore[import-untyped]
from grave_image_matching.constants import TAFEL_PATTERN
from grave_image_matching.extract_boxes import extract_from_image
from grave_image_matching.logger import logger

PDF_PATH = "grave_image_matching/data/katalog2_tafeln.pdf"
OUTPUT_FOLDER = "grave_image_matching/katalog2/images"
OUTPUT_TAFEL_SPLITS_FOLDER = "grave_image_matching/katalog2/images/tafel-splits"


def extract_images_and_save_json(
    pdf_path: str, output_folder: str
) -> dict[str, Optional[str]]:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(2.0, 2.0)

    captions_data: dict[str, str | None] = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        logger.info("Processing page", page=page_num)

        page_height = page.rect.height
        images: list[dict[str, Any]] = page.get_image_info(hashes=False)  # type: ignore[attr-defined]
        dict_text: dict[str, Any] = page.get_text("dict")

        all_lines: list[dict[str, Any]] = []
        for block in dict_text["blocks"]:
            block_dict: dict[str, Any] = block
            if block_dict["type"] == 0:
                for line in block_dict["lines"]:
                    line_dict: dict[str, Any] = line
                    x0 = min([s["bbox"][0] for s in line_dict["spans"]])
                    y0 = min([s["bbox"][1] for s in line_dict["spans"]])
                    x1 = max([s["bbox"][2] for s in line_dict["spans"]])
                    y1 = max([s["bbox"][3] for s in line_dict["spans"]])

                    text = " ".join([s["text"] for s in line_dict["spans"]]).strip()

                    all_lines.append({"bbox": fitz.Rect(x0, y0, x1, y1), "text": text})

        for i, img in enumerate(images):
            img_dict: dict[str, Any] = img
            img_bbox = fitz.Rect(img_dict["bbox"])

            image_filename = f"page{page_num}_img{i}.png"
            image_path = os.path.join(output_folder, image_filename)

            logger.info("Saving image", index=i, filename=image_filename)
            pix = page.get_pixmap(matrix=mat, clip=img_bbox, alpha=False)
            pix.save(image_path)  # type: ignore[attr-defined]

            best_caption: str | None = None
            min_score = 10000

            img_center_y = (img_bbox.y0 + img_bbox.y1) / 2

            for line in all_lines:
                line_bbox: fitz.Rect = line["bbox"]
                line_text: str = line["text"]

                debug_text = (
                    (line_text[:30] + "..") if len(line_text) > 30 else line_text
                )

                if not line_text:
                    continue

                if line_bbox.y1 < img_center_y:
                    continue

                # filter 2: Horizontal Center Alignment
                line_center_x = (line_bbox.x0 + line_bbox.x1) / 2
                if line_center_x < (img_bbox.x0 - 50) or line_center_x > (
                    img_bbox.x1 + 50
                ):
                    logger.debug(
                        "Skipping line (in image)",
                        text=debug_text,
                        line_y0=f"{line_bbox.y0:.1f}",
                        img_y1=f"{img_bbox.y1:.1f}",
                    )
                    continue

                logger.debug(
                    "Checking text",
                    text=line_text[:20],
                    img_y1=img_bbox.y1,
                    line_y0=line_bbox.y0,
                    img_center_y=img_center_y,
                    line_y1=line_bbox.y1,
                )

                # filter 1 Check
                if line_bbox.y1 < img_center_y:
                    logger.debug("Skipped (too far above image)")
                    continue

                # filter 3: page numbers
                if (
                    line_text.replace(" ", "").isdigit()
                    and line_bbox.y0 > page_height * 0.95
                ):
                    logger.debug("Skipping page number", text=debug_text)
                    continue

                # scoring
                v_dist = line_bbox.y0 - img_bbox.y1
                is_abb = bool(re.search(r"Abb[\. ]", line_text, re.IGNORECASE))

                score = v_dist

                if is_abb:
                    score -= 5000

                logger.debug("Caption candidate", score=f"{score:.1f}", text=debug_text)

                if score < min_score:
                    min_score = score
                    best_caption = line_text

            if best_caption:
                clean_caption = " ".join(best_caption.split())
                logger.info("Matched caption", caption=clean_caption)
                captions_data[image_filename] = clean_caption
            else:
                logger.debug("No caption found")
                captions_data[image_filename] = None

    doc.close()

    # save dictionary to JSON file
    json_path = os.path.join(output_folder, "captions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(captions_data, f, indent=4, ensure_ascii=False)

    logger.info("Saved JSON caption map", path=str(json_path))

    return captions_data


if __name__ == "__main__":
    captions_map = extract_images_and_save_json(PDF_PATH, OUTPUT_FOLDER)

    logger.info("Post-processing Tafeln")

    for filename, caption in captions_map.items():
        # if tafel -> split into subimages
        if caption and TAFEL_PATTERN.search(caption):
            full_image_path = os.path.join(OUTPUT_FOLDER, filename)
            image_base_name = os.path.splitext(filename)[0]
            image_output_dir = os.path.join(OUTPUT_TAFEL_SPLITS_FOLDER, image_base_name)

            if not os.path.exists(image_output_dir):
                os.makedirs(image_output_dir)

            extract_from_image(full_image_path, image_output_dir)
