import csv
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, Optional, cast

import pandas as pd
from openai import OpenAI
from openinference.instrumentation import using_attributes

from grave_image_matching.constants import GRAVE_NAME_COL
from grave_image_matching.logger import logger
from grave_image_matching.tafel_matching_strategies.base import TafelBoxToGraveMatcher
from grave_image_matching.tafel_matching_strategies.mllm.mllm_request_strategies import (
    MLLMRequestStrategy,
)
from grave_image_matching.utils import build_candidate_dict

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

MLLM_PREDICTIONS_CSV_BASE = Path("grave_image_matching/output/mllm_predictions")
MLLM_PREDICTIONS_CSV = (
    MLLM_PREDICTIONS_CSV_BASE.parent
    / f"{MLLM_PREDICTIONS_CSV_BASE.name}_{timestamp}.csv"
)


class TafelBoxToGraveMLLMMatcher(TafelBoxToGraveMatcher):
    """
    Concrete Strategy: Matches using OpenAI MLLM.
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        graves_df: pd.DataFrame,
        mllm_request_strategy: MLLMRequestStrategy,
    ):
        super().__init__(graves_df)
        self.client = client
        self.model_name = model_name
        self.mllm_request_strategy = mllm_request_strategy

    def match(
        self,
        subimage_path: Path,
        candidate_graves: pd.DataFrame,
        tafel_id: str,
        tafel_image_path: Path,
        tafel_caption: str,
    ) -> Optional[int]:
        logger.debug("Calling MLLM to match subimage", subimage=str(subimage_path))

        if candidate_graves.empty:
            return None

        candidate_dict = build_candidate_dict(candidate_graves, tafel_id)

        if not candidate_dict:
            # If no graves specifically reference this Tafel ID number, skip MLLM
            return None

        prompt_text, base64_images = self.mllm_request_strategy.prepare_payload(
            subimage_path, tafel_image_path, tafel_caption, candidate_dict
        )

        with using_attributes(
            metadata={
                "tafel_id": tafel_id,
                "tafel_caption": tafel_caption,
                "sub_image": str(subimage_path),
                "tafel_image": str(tafel_image_path),
            },
        ):
            result_text = self._call_mllm_api(prompt_text, base64_images)

        matched_idx = self._parse_mllm_result(result_text, candidate_graves)

        self._save_prediction(
            tafel_id,
            tafel_image_path,
            subimage_path,
            result_text,
            matched_idx,
            tafel_caption,
        )

        return matched_idx

    def _build_prompt(self, tafel_caption: str, candidate_dict: dict) -> str:
        candidates_text_lines = [
            f'[{idx}] "{grave_name}": "{", ".join(dict.fromkeys(refs))}"'
            for idx, (grave_name, refs) in candidate_dict.items()
        ]
        candidate_block = "\n".join(candidates_text_lines)

        return f"""Du erhälst ein Bild einer Tafel von archäologischen Fundstücken eines Grabkataloges und zusätzlich ein ausgeschnittes Subbild aus dieser Tafel. Das Subbild gehört zu einem der folgenden Gräber. 
Im Subbild befinden sich Textannotationen (oft Buchstaben wie A, B oder Nummern), die sich auf den erwähnten Tafeleintrag vom Grab beziehen.
Die Tafelreferenzen sind meistens wie folgt aufgebaut: "Taf. X-N/n". Dabei ist X die Tafelnummer (irrelevant für dich). Die Kombination aus N und n bezieht sich auf das Subbild. Dabei steht N meistens in der linken oberen Ecke und n in der linken unteren Ecke. 
Anhand dieser Kennzeichen kannst du ein Grab zuordnen.
Findest du keine solche Annotationen oder kannst kein Grab eindeutig zuordnen, gib "null" zurück.
Gib nur den exakten Index des passenden Grabes zurück (ohne Satzzeichen oder Erklärung, nur die Zahl).

Das ist die Caption der Gesamttafel (nicht des gezeigten Subbildes): 
"{tafel_caption}"

Folgend sind die möglichen Gräber-Indizes angegeben mit den Bezeichungen ihrer zugehörenden Subbilder / Fundstücken (Schema: [Grab-Index] "Grab-Titel": "Bez. 1, Bez. 2, ..."):

{candidate_block}"""

    def _call_mllm_api(
        self, prompt_text: str, base64_images: list[str]
    ) -> Optional[str]:
        try:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
            for img_b64 in base64_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    }
                )

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=cast(
                    Any,
                    [
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                ),
                max_tokens=50,
                temperature=0.0,
            )

            if "academiccloud" in str(self.client.base_url):
                sleep(18)  # prevent rate limits

            response_content = response.choices[0].message.content
            return response_content.strip() if response_content else None
        except Exception as e:
            logger.error("MLLM API request failed", error=str(e))
            return None

    def _parse_mllm_result(
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
                logger.debug("MLLM returned index", index=returned_index)
                return returned_index
            else:
                logger.debug(
                    "MLLM returned index not in candidate list", result=clean_result
                )
                return None
        except ValueError:
            logger.debug("MLLM returned invalid integer", result=clean_result)
            return None

    def _save_prediction(
        self,
        tafel_id: str,
        tafel_image_path: Path,
        subimage_path: Path,
        mllm_response: Optional[str],
        matched_grave_index: Optional[int],
        tafel_caption: str,
    ) -> None:
        """Appends a MLLM prediction to the CSV file. Uses internal graves_df state."""
        grave_name = ""

        # Access the DataFrame stored in the class instance
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
            "mllm_response": mllm_response if mllm_response else "",
            "model": self.model_name,
        }

        df_row = pd.DataFrame([row_data])
        file_exists = MLLM_PREDICTIONS_CSV.exists()
        df_row.to_csv(
            MLLM_PREDICTIONS_CSV,
            mode="a",
            header=not file_exists,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL,
        )
