from typing import Literal, Optional, TypedDict

ModelProvider = Literal["openai", "gwdg"]


class Header(TypedDict):
    id: int
    header_text: str
    page_id: int
    heading_level: Optional[int]
    polygon: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]


class SectionData(TypedDict):
    id: int
    title: str
    heading_level: Optional[int]
    page_id: int
    text: Optional[str]


class TextBlock(TypedDict):
    page: int
    text: str
    y_top: float
    y_bottom: float
