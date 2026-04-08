from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from grave_image_matching.utils import encode_image


class MLLMRequestStrategy(ABC):
    """
    Defines how to prepare the payload (text prompt + images) for the MLLM.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        pass

    @abstractmethod
    def prepare_payload(
        self,
        subimage_path: Path,
        tafel_image_path: Path,
        tafel_caption: str,
        candidate_dict: dict[Any, tuple[str, list[str]]],
    ) -> tuple[str, list[str]]:
        """
        Returns:
            Tuple[str, List[str]]: (The Prompt Text, List of Base64 Image Strings)
        """
        pass


class FullTafelContextStrategy(MLLMRequestStrategy):
    """
    Strategy: Sends BOTH the full Tafel image and the Subimage.
    """

    @property
    def strategy_name(self) -> str:
        return "FullTafelContext"

    def prepare_payload(
        self, subimage_path, tafel_image_path, tafel_caption, candidate_dict
    ):
        base64_imgs = [encode_image(subimage_path), encode_image(tafel_image_path)]

        candidates_text_lines = [
            f'[{idx}] "{grave_name}": "{", ".join(dict.fromkeys(refs))}"'
            for idx, (grave_name, refs) in candidate_dict.items()
        ]
        candidate_block = "\n".join(candidates_text_lines)

        prompt = f"""Du erhälst ein Bild einer Tafel von archäologischen Fundstücken eines Grabkataloges und zusätzlich ein ausgeschnittes Subbild aus dieser Tafel. Das Subbild gehört zu einem der folgenden Gräber. 
Im Subbild befinden sich Textannotationen (oft Buchstaben wie A, B oder Nummern), die sich auf den erwähnten Tafeleintrag vom Grab beziehen.
Die Tafelreferenzen sind meistens wie folgt aufgebaut: "Taf. X-N/n". Dabei ist X die Tafelnummer (irrelevant für dich). Die Kombination aus N und n bezieht sich auf das Subbild. Dabei steht N meistens in der linken oberen Ecke und n in der linken unteren Ecke. 
Anhand dieser Kennzeichen kannst du ein Grab zuordnen.
Findest du keine solche Annotationen oder kannst kein Grab eindeutig zuordnen, gib "null" zurück.
Gib nur den exakten Index des passenden Grabes zurück (ohne Satzzeichen oder Erklärung, nur die Zahl).

Das ist die Caption der Gesamttafel (nicht des gezeigten Subbildes): 
"{tafel_caption}"

Folgend sind die möglichen Gräber-Indizes angegeben mit den Bezeichungen ihrer zugehörenden Subbilder / Fundstücken (Schema: [Grab-Index] "Grab-Titel": "Bez. 1, Bez. 2, ..."):

{candidate_block}"""

        return prompt, base64_imgs


class SubImageOnlyStrategy(MLLMRequestStrategy):
    """
    Strategy: Sends ONLY the Subimage.
    """

    @property
    def strategy_name(self) -> str:
        return "SubImageOnly"

    def prepare_payload(
        self, subimage_path, tafel_image_path, tafel_caption, candidate_dict
    ):
        base64_imgs = [encode_image(subimage_path)]

        candidates_text_lines = [
            f'[{idx}] "{grave_name}": "{", ".join(dict.fromkeys(refs))}"'
            for idx, (grave_name, refs) in candidate_dict.items()
        ]
        candidate_block = "\n".join(candidates_text_lines)

        prompt = f"""Du erhälst ein ausgeschnittes Subbild aus einer Tafel eines archäologischen Grabkataloges. Dieses Subbild gehört zu einem der folgenden Gräber. 
Im Subbild befinden sich Textannotationen (oft Buchstaben wie A, B oder Nummern), die sich auf den erwähnten Tafeleintrag vom Grab beziehen.
Die Tafelreferenzen sind meistens wie folgt aufgebaut: "Taf. X-N/n". Dabei ist X die Tafelnummer (irrelevant für dich). Die Kombination aus N und n bezieht sich auf das Subbild. Dabei steht N meistens in der linken oberen Ecke und n in der linken unteren Ecke. 
Anhand dieser Kennzeichen kannst du ein Grab zuordnen. Ignoriere alle Kennzeichen die sich an den Fundstücken befinden.
Findest du keine solche Annotationen oder kannst kein Grab eindeutig zuordnen, gib "null" zurück.
Gib nur den exakten Index des passenden Grabes zurück (ohne Satzzeichen oder Erklärung, nur die Zahl).

Folgend sind die möglichen Gräber-Indizes angegeben mit den Bezeichungen ihrer zugehörenden Subbilder / Fundstücken (Schema: [Grab-Index] "Grab-Titel": "Bez. 1, Bez. 2, ..."):

{candidate_block}"""

        return prompt, base64_imgs
