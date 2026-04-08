from abc import ABC, abstractmethod
from typing import Any, cast

import fitz

from grave_extraction.models import Header, SectionData, TextBlock


class SectionsBuilder(ABC):
    @abstractmethod
    def build_sections(self, pdf_path: str, headers: list[Header]) -> list[SectionData]:
        """
        Build document sections for a given PDF and its headers.
        Implementations can use different algorithms while sharing this interface.
        """
        raise NotImplementedError


def _extract_text_blocks_with_coords(pdf_path: str) -> list[TextBlock]:
    """
    Extracts text line by line from a PDF for granular accuracy.
    """
    doc = fitz.open(pdf_path)
    all_lines = []
    for page_idx, page in enumerate(cast(list[fitz.Page], doc)):
        page_dict = cast(dict[str, Any], page.get_text("dict"))
        for block in page_dict.get("blocks", []):
            if block.get("type") == 0:  # 0 indicates a text block
                for line in block.get("lines", []):
                    text = " ".join(
                        [span["text"] for span in line.get("spans", [])]
                    ).strip()
                    if not text:
                        continue

                    bbox = line["bbox"]
                    all_lines.append(
                        TextBlock(
                            page=page_idx,
                            text=text,
                            y_top=float(bbox[1]),
                            y_bottom=float(bbox[3]),
                        )
                    )
    doc.close()
    return all_lines


class HybridSectionsBuilder(SectionsBuilder):
    """
    Default implementation of `SectionsBuilder` that uses the current
    hybrid geometric and textual algorithm to determine section boundaries.
    """

    def build_sections(
        self,
        pdf_path: str,
        headers: list[Header],
    ) -> list[SectionData]:
        text_blocks = _extract_text_blocks_with_coords(pdf_path)

        processed_headers: list[dict[str, Any]] = []
        for i, header in enumerate(headers):
            if not header.get("header_text"):
                continue

            if not header.get("polygon"):
                raise ValueError(f"No polygon info for header {header['id']}")

            y_coords = [p[1] for p in header["polygon"]]
            processed_headers.append(
                {
                    "original_toc_index": i,
                    "title": header["header_text"],
                    "page_id": header["page_id"],
                    "y_top": min(y_coords),
                    "y_bottom": max(y_coords),
                }
            )

        processed_headers.sort(key=lambda h: (h["page_id"], h["y_top"]))
        section_texts: dict[int, list[str]] = {i: [] for i in range(len(headers))}

        for i, current_header in enumerate(processed_headers):
            start_page = current_header["page_id"]
            start_y = current_header["y_bottom"]

            end_page, end_y = (float("inf"), float("inf"))
            next_header_title_norm = None

            if i + 1 < len(processed_headers):
                next_header = processed_headers[i + 1]
                end_page = next_header["page_id"]
                end_y = next_header["y_top"]
                # Pre-normalize the next header's title for efficient matching
                next_header_title_norm = " ".join(next_header["title"].split()).lower()

            for block in text_blocks:
                is_after_start = (block["page"] > start_page) or (
                    block["page"] == start_page and block["y_top"] >= start_y
                )

                is_before_end = (block["page"] < end_page) or (
                    block["page"] == end_page and block["y_top"] < end_y
                )

                if is_after_start and is_before_end:
                    # Before assigning, perform a final check to ensure this block is not
                    # the title of the next section, which can happen with small coordinate mismatches.
                    if next_header_title_norm:
                        block_text_norm = " ".join(block["text"].split()).lower()
                        # If text matches the next header title, we must exclude it.
                        if block_text_norm == next_header_title_norm:
                            continue  # Skip this block; it's the start of the next section.

                    # If all checks pass, assign the block to the current section.
                    toc_idx = current_header["original_toc_index"]
                    section_texts[toc_idx].append(block["text"])

        final_sections: list[SectionData] = []
        for i, header_data in enumerate(headers):
            text = "\n".join(section_texts.get(i, [])).strip()
            final_sections.append(
                SectionData(
                    id=header_data["id"],
                    title=header_data["header_text"],
                    heading_level=header_data["heading_level"],
                    page_id=header_data["page_id"],
                    text=text or None,
                )
            )

        return final_sections
