import csv
import json
import os
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Optional

import cv2
import easyocr  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytesseract  # type: ignore[import-untyped]
from detectron2 import model_zoo  # type: ignore[import-untyped]
from detectron2.config import get_cfg  # type: ignore[import-untyped]
from detectron2.engine import DefaultPredictor  # type: ignore[import-untyped]
from grave_image_matching.constants import GRAVE_NAME_COL
from grave_image_matching.logger import logger
from grave_image_matching.tafel_matching_strategies.base import TafelBoxToGraveMatcher
from grave_image_matching.utils import build_candidate_dict
from openai import OpenAI
from openinference.instrumentation import using_attributes
from PIL import Image, ImageOps

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

LLM_PREDICTIONS_CSV_BASE = Path("grave_image_matching/output/id_llm_predictions")
LLM_PREDICTIONS_CSV = (
    LLM_PREDICTIONS_CSV_BASE.parent / f"{LLM_PREDICTIONS_CSV_BASE.name}_{timestamp}.csv"
)

TESSERACT_CONFIG = r"--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"


class TafelBoxToGraveWithIdExtractionAndLLMMatcher(TafelBoxToGraveMatcher):
    """
    Concrete Strategy: Matches by extracting an ID via Faster R-CNN + EasyOCR,
    then asking an LLM to match the extracted text to candidate graves.
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        graves_df: pd.DataFrame,
        id_model_config_dir: str = "tafel_subbox_id_extraction_model/output",
        score_threshold: float = 0.7,
        device: str = "cpu",
    ):
        super().__init__(graves_df)
        self.client = client
        self.model_name = model_name

        logger.info("Loading ID Extraction Model")
        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        )
        config_path = os.path.join(id_model_config_dir, "config.yaml")
        if os.path.exists(config_path):
            self.cfg.merge_from_file(config_path)

        self.cfg.MODEL.WEIGHTS = os.path.join(id_model_config_dir, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.MODEL.DEVICE = device

        self.predictor = DefaultPredictor(self.cfg)

        logger.info("Loading EasyOCR")
        self.reader = easyocr.Reader(["de", "en"], gpu=(device != "cpu"))

    def match(
        self,
        subimage_path: Path,
        candidate_graves: pd.DataFrame,
        tafel_id: str,
        tafel_image_path: Path,
        tafel_caption: str,
    ) -> Optional[int]:
        logger.debug("Processing with ID Extraction + LLM", subimage=str(subimage_path))

        if candidate_graves.empty:
            return None

        # filter candidates specific to this tafel ID
        candidate_dict = build_candidate_dict(candidate_graves, tafel_id)
        if not candidate_dict:
            return None

        # extract ID image and perform OCR
        extracted_text = self._extract_id_text(subimage_path)
        logger.debug("Extracted text", subimage=subimage_path.name, text=extracted_text)

        # call LLM
        with using_attributes(
            metadata={
                "tafel_id": tafel_id,
                "sub_image": str(subimage_path),
                "extracted_text": extracted_text,
            },
        ):
            prompt_text = self._build_prompt(
                tafel_caption, candidate_dict, extracted_text
            )
            result_text = self._call_llm_api(prompt_text)

        matched_idx = self._parse_llm_result(result_text, candidate_graves)

        # save results
        self._save_prediction(
            tafel_id,
            tafel_image_path,
            subimage_path,
            result_text,
            matched_idx,
            tafel_caption,
            extracted_text,
        )

        return matched_idx

    def _extract_id_text(self, subimage_path: Path) -> str:
        """
        Loads image, detects ID box, crops it, saves crop to disk, runs OCR,
        and saves a JSON report of the results.
        """
        image_path_str = str(subimage_path)
        img = cv2.imread(image_path_str)
        if img is None:
            logger.error("Could not read image", path=image_path_str)
            return ""

        # Detectron2 Inference
        outputs = self.predictor(img)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()

        if len(boxes) == 0:
            return ""

        detected_texts = []
        ocr_metadata = []

        # Create folder for debug crops and JSON
        save_dir = subimage_path.parent / subimage_path.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            h, w, _ = img.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop_img_bgr = img[y1:y2, x1:x2]

            if crop_img_bgr.size == 0:
                continue

            # Convert OpenCV (BGR Array) to PIL (RGB Object)
            crop_img_rgb = cv2.cvtColor(crop_img_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(crop_img_rgb)

            # Preprocessing (Using PIL)
            resized_image = ImageOps.expand(pil_image, border=20, fill="white")
            resized_image = resized_image.resize(
                (resized_image.width * 2, resized_image.height * 2),
                Image.Resampling.LANCZOS,
            )

            # Save the crop
            save_filename = f"extracted_id_{i}.png"
            try:
                resized_image.save(save_dir / save_filename)
            except Exception as e:
                logger.warning("Failed to save crop", error=str(e))

            box_text_found = []
            used_model = "None"

            # EasyOCR
            try:
                ocr_results = self.reader.readtext(np.array(resized_image), detail=0)
                if ocr_results:
                    box_text_found.extend(ocr_results)
                    used_model = "EasyOCR"
            except Exception as e:
                logger.warning(
                    "EasyOCR failed", box=i, subimage=subimage_path.name, error=str(e)
                )

            # Tesseract
            if not box_text_found:
                try:
                    # Try reading as a block
                    text = pytesseract.image_to_string(
                        resized_image, config="--psm 6"
                    ).strip()

                    if not text:
                        # Fallback to Single Character
                        text = pytesseract.image_to_string(
                            resized_image, config="--psm 10"
                        ).strip()

                    if text:
                        box_text_found.append(text)
                        used_model = "Tesseract"
                except Exception as e:
                    logger.warning(
                        "Tesseract failed",
                        box=i,
                        subimage=subimage_path.name,
                        error=str(e),
                    )

            # Consolidate results for this box
            if box_text_found:
                final_text = " ".join(box_text_found)
                detected_texts.append(final_text)

                # Add to metadata list
                ocr_metadata.append(
                    {
                        "image_file": save_filename,
                        "extracted_text": final_text,
                        "model_used": used_model,
                    }
                )
            else:
                ocr_metadata.append(
                    {
                        "image_file": save_filename,
                        "extracted_text": "",
                        "model_used": "Failed",
                    }
                )

        # Save Metadata to JSON
        json_path = save_dir / "ocr_result.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(ocr_metadata, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.warning("Failed to save JSON report", error=str(e))

        return " ".join(detected_texts).strip()

    def _build_prompt(
        self, tafel_caption: str, candidate_dict: dict, extracted_text: str
    ) -> str:
        candidates_text_lines = [
            f'[{idx}] "{grave_name}": "{", ".join(dict.fromkeys(refs))}"'
            for idx, (grave_name, refs) in candidate_dict.items()
        ]
        candidate_block = "\n".join(candidates_text_lines)

        return f"""Du bist ein präziser, skeptischer Klassifikations-Agent für archäologische Metadaten.
Du erhälst Daten zu einem Subbild von einer archäologischen Tafel.
Aus dem Subbild wurde mittels OCR folgender Text extrahiert, welcher die Kennzeichnung/ID des Subbildes angibt und vom Grab referenziert wird. 

Die Tafelreferenzen sind meistens wie folgt aufgebaut: "Taf. X-N/n". 
- X ist die Tafelnummer.
- N/n ist die extrahierte Nummerierung auf dem Subbild (z.B. 1, 2, A, B).

Das Subbild gehört zu einem der folgenden Gräber.

# Aufgabe:
- Vergleiche den OCR-Text streng mit den Referenzen der Kandidaten.
- Ordne das passende Grab zu und gib den Index des Grabes zurück.
- Wenn du kein Grab zuordnen kannst, gib null zurück.

Gib nur den Index zurück, nichts anderes.

# Beispiel 1:

Kontext: Tafel-Caption: "Taf. 2. Abrahám I, Gräber 7, 8, 12 – 15, 18 (7: 1 – 4; 12: 2 – Knochen; 7: 5; 13: 1 – Stein; 14: 1; 15: 1 – Bernstein; 18: 1, 6 – Mollusken)."

Extrahierter OCR-Text aus dem Sibbild: "14"

Kandidaten (Format: [Index] "Grabname": "Referenzen"):
[9] "Grab 7 (G 3)": "Taf.2-7:1, Taf.2-7:2, Taf.2-7:3, Taf.2-7:4, Taf.2-7:5"
[10] "Grab 8 (F 3)": "Taf.2-8:1"
[14] "Grab 12 (H 4)": "Taf.2-12:1, Taf.2-12:2"
[15] "Grab 13 (H 6)": "Taf.2-13:1"
[16] "Grab 14 (nicht eingezeichnet)": "Taf.2-14:1, Taf.2-14:2, Taf.2-14:3"
[17] "Grab 15 (E 5)": "Taf.2-15:1, Taf.2-15:2, Taf.2-15:3"
[20] "Grab 18 (F 5)": "Taf.2-18:1, Taf.2-18:2, Taf.2-18:3, Taf.2-18:4, Taf.2-18:5, Taf.2-18:6, Taf.2-18:7"

Korrekte Antwort: "16" 
(weil "14" Referenz auf "Taf.2-14" -> Grab 14 -> Index 16)

# Beispiel 2:

Kontext: Tafel-Caption: "Taf. 107. A – Voderady-Slovenská Nová Ves, Gräber 2, 3; B – Vozokany I – IV, VII – Grabzusammenhang unbekannt (A/2: 1 – 3;"

Extrahierter OCR-Text aus dem Subbild: "2 A"

Kandidaten (Format: [Index] "Grabname": "Referenzen"):
[651] "Grab 2": "Taf.107-A/2:1, Taf.107-A/2:2,3"
[652] "Grab 3": "Taf.107-A/3:1, Taf.107-A/3:2, Taf.107-A/3:3"

Korrekte Antwort: "651"
(weil "2 A" Referenz auf "Taf.107-A/2" -> Grab 2 -> Index 651)

# Eingabe

Kontext: Tafel-Caption: "{tafel_caption}"

Extrahierter OCR-Text aus dem Subbild: "{extracted_text}"

Kandidaten (Format: [Index] "Grabname": "Referenzen"):
{candidate_block}"""

    def _call_llm_api(self, prompt_text: str) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=10,
                temperature=0.0,
            )

            if "academiccloud" in str(self.client.base_url):
                sleep(18)

            return response.choices[0].message.content
        except Exception as e:
            logger.error("LLM API request failed", error=str(e))
            return None

    def _parse_llm_result(
        self, result_text: Optional[str], candidate_graves: pd.DataFrame
    ) -> Optional[int]:
        if not result_text:
            return None

        clean_result = result_text.strip().strip('"').strip("'")
        if clean_result.lower() in ["null", "none", "no match"]:
            return None

        try:
            returned_index = int(clean_result)
            if returned_index in candidate_graves.index:
                return returned_index
        except ValueError:
            logger.error("Can't parse LLM response", result=clean_result)

        return None

    def _save_prediction(
        self,
        tafel_id: str,
        tafel_image_path: Path,
        subimage_path: Path,
        llm_response: Optional[str],
        matched_grave_index: Optional[int],
        tafel_caption: str,
        extracted_text: str,
    ) -> None:
        """Appends prediction to the CSV file."""
        grave_name = ""
        if (
            matched_grave_index is not None
            and matched_grave_index in self.graves_df.index
        ):
            grave_name = str(self.graves_df.at[matched_grave_index, GRAVE_NAME_COL])

        row_data = {
            "tafel_id": tafel_id,
            "tafel_image_path": str(tafel_image_path),
            "subimage_path": str(subimage_path),
            "matched_grave_name": grave_name,
            "matched_grave_index": matched_grave_index
            if matched_grave_index is not None
            else "",
            "tafel_caption": tafel_caption,
            "extracted_id_text": extracted_text,
            "llm_response": llm_response if llm_response else "",
            "model": self.model_name,
        }

        df_row = pd.DataFrame([row_data])
        file_exists = LLM_PREDICTIONS_CSV.exists()

        LLM_PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)

        df_row.to_csv(
            LLM_PREDICTIONS_CSV,
            mode="a",
            header=not file_exists,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL,
        )
