from dataclasses import dataclass, field
from typing import Optional

from grave_extraction.extraction.grave_record import GraveRecord
from grave_extraction.models import SectionData


class GraveRecordWithPageIdAndSectionTitle(GraveRecord):
    page_id: int
    section_title: str


@dataclass
class ExtractionState:
    """Handles control flow state (where we are, what we've processed, what we've extracted)."""

    section_buffer: Optional[SectionData] = None
    """Temporary data for the currently processed section. This is cleared after each section is processed."""
    section_path: list[str] = field(default_factory=list)
    """Hierarchical list of section titles, e.g. `['3.1 Abrahám', 'Abrahám I', 'Grab 3']`"""
    extracted_records: list[GraveRecordWithPageIdAndSectionTitle] = field(
        default_factory=list
    )
    """Accumulated list of all extracted grave records (flattened dicts for CSV output)."""

    _base_level_offset: Optional[int] = field(default=None, init=False, repr=False)
    """Internal: remembers whether we're using 0-based or 1-based levels."""

    def push_section(self, section_data: SectionData):
        """
        Update the current hierarchical path (truncate higher levels and replace same-level nodes).
        Supports both 0-based (root=0) and 1-based (root=1) hierarchies.
        """
        if section_data["heading_level"] is None:
            raise ValueError("No heading_level defined!")

        # Detect base level convention on first call
        if self._base_level_offset is None:
            # If first section is level 0 → offset = 0, else offset = 1
            self._base_level_offset = 0 if section_data["heading_level"] == 0 else 1

        # Normalize to 0-based internally
        normalized_level = max(
            section_data["heading_level"] - self._base_level_offset, 0
        )

        self.section_path = self.section_path[:normalized_level]
        self.section_path.append(section_data["title"])

    def clear_buffer(self):
        self.section_buffer = None
