from grave_extraction.logger import logger
from grave_extraction.models import Header
from grave_extraction.utils import normalize_text


def calculate_heading_level_accuracy(
    gt_headers: list[Header], comparison_headers: list[Header]
) -> float:
    """
    Calculates the accuracy of heading levels between two lists of headers.

    The function compares headers element-wise based on their position in the list.
    A comparison is considered a "match" only if both the header_text and the
    heading_level are identical.

    Args:
        gt_headers: The first list of Header objects.
        comparison_headers: The second list of Header objects to compare against.

    Returns:
        The accuracy as a float between 0.0 and 1.0.
    """
    if not gt_headers and not comparison_headers:
        return 1.0
    if not gt_headers or not comparison_headers:
        return 0.0

    # If lists have different lengths, it indicates missing or extra headers. We'll compare up to the length of the shorter list.
    num_comparisons = min(len(gt_headers), len(comparison_headers))
    if len(gt_headers) != len(comparison_headers):
        logger.error(
            f"Warning: Header lists have different lengths ({len(gt_headers)} vs {len(comparison_headers)}). "
            f"Comparing the first {num_comparisons} elements.",
        )

    correct_matches = 0
    for i in range(num_comparisons):
        header1 = gt_headers[i]
        header2 = comparison_headers[i]

        if (
            normalize_text(header1["header_text"])
            == normalize_text(header2["header_text"])
            and header1["heading_level"] == header2["heading_level"]
        ):
            correct_matches += 1

    return correct_matches / num_comparisons
