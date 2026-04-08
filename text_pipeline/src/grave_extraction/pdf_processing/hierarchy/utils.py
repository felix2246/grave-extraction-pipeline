import re
from typing import Optional, Tuple

from grave_extraction.models import Header


class NumberingParser:
    """
    Centralized logic for parsing structured numbering (1.1, IV., etc.).
    """

    @staticmethod
    def parse(header: Header) -> Optional[Tuple[str, int]]:
        """
        Checks if a header starts with numbering. Returns (number_string, zero_based_level) if match found, else None.
        """
        text = header["header_text"].strip()
        if not text:
            return None

        tokens = text.split(maxsplit=1)
        if not tokens:
            return None

        number_part = tokens[0]

        # Arabic numbering (e.g., "1.", "2.1.")
        # Matches "1", "1.", "1.1", "1.1."
        arabic_match = re.match(r"^(\d{1,3}(\.\d{1,3})*)\.?$", number_part)
        if arabic_match:
            number_str = arabic_match.group(1)
            # Level is determined by dots: "1" -> 0, "1.1" -> 1
            level = len(number_str.split(".")) - 1
            return number_str, level

        # Roman numeral numbering (Assumed Level 0)
        roman_match = re.match(
            r"^(?=[MDCLXVI])M*(C[MD]|D?C*)(X[CL]|L?X*)(I[XV]|V?I*)\.?$",
            number_part,
            re.IGNORECASE,
        )
        if roman_match:
            return roman_match.group(0), 0

        return None


def assign_levels_from_cluster_labels(cluster_labels: list[int]) -> list[int]:
    """
    Converts a sequence of cluster IDs (e.g., [0, 0, 1, 2, 1]) into relative indentation levels based on stack logic.
    """
    if not cluster_labels:
        return []

    levels: list[int] = []
    cluster_stack: list[int] = []
    current_level = -1

    for cluster_id in cluster_labels:
        if not cluster_stack or cluster_id not in cluster_stack:
            current_level += 1
            cluster_stack.append(cluster_id)
        else:
            # Pop until we find the cluster_id again
            while cluster_stack and cluster_stack[-1] != cluster_id:
                cluster_stack.pop()
                current_level -= 1

        levels.append(max(0, current_level))

    return levels
