from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd


class TafelBoxToGraveMatcher(ABC):
    """
    Abstract Strategy for matching a subimage to a specific grave record.
    """

    def __init__(self, graves_df: pd.DataFrame):
        self.graves_df = graves_df

    @abstractmethod
    def match(
        self,
        subimage_path: Path,
        candidate_graves: pd.DataFrame,
        tafel_id: str,
        tafel_image_path: Path,
        tafel_caption: str,
    ) -> Optional[int]:
        """
        Execute the matching strategy.
        Returns the index of the matched grave in the original dataframe, or None.
        """
        pass
