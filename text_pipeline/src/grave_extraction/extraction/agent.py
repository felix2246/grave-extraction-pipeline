import ast
import time
from typing import Any

import pandas as pd
from langchain_core.language_models import BaseChatModel
from openinference.instrumentation import using_attributes
from pydantic import BaseModel

from grave_extraction.extraction.context_manager import (
    ContextManager,
    StructuredGraveContext,
)
from grave_extraction.extraction.grave_record import GraveRecord
from grave_extraction.extraction.state import (
    ExtractionState,
    GraveRecordWithPageIdAndSectionTitle,
)
from grave_extraction.logger import logger
from grave_extraction.models import SectionData
from grave_extraction.prompt_strategy import PromptStrategy
from grave_extraction.utils import timed


class SharedContext(BaseModel):
    structured: StructuredGraveContext
    interpretive: list[str]


class CombinedSectionOutput(BaseModel):
    shared_context: SharedContext
    graves: list[GraveRecord]


def extract_from_section(
    llm: BaseChatModel,
    state: ExtractionState,
    context_manager: ContextManager,
    grave_extraction_prompt_strategy: PromptStrategy,
) -> ExtractionState:
    """
    Extracts general context (if present) and the graves of a section in a single LLM call.
    """
    if not state.section_buffer:
        raise Exception("No section buffer for this state!")

    section = state.section_buffer

    if section["heading_level"] is None:
        raise Exception(f"No heading_level set for section {section['id']}")

    hierarchy_str = " > ".join(state.section_path)

    # remove invalid context from same level
    context_manager.pop_to_level(section["heading_level"])

    logger.info("Hierarchy:", hierarchy_str=hierarchy_str)

    inherited_structured = context_manager.merged_structured()
    inherited_interpretive = context_manager.merged_interpretive()
    logger.info(
        "Context:",
        inherited_structured=inherited_structured.model_dump(
            exclude_none=True, exclude_unset=True
        ),
        inherited_interpretive=inherited_interpretive,
    )

    inherited_structured_dict = inherited_structured.model_dump(
        exclude_none=True, exclude_defaults=True
    )
    inherited_structured_model_dump = (
        inherited_structured.model_dump_json(exclude_none=True, exclude_defaults=True)
        if inherited_structured_dict != {}
        else None
    )

    prompt = grave_extraction_prompt_strategy.format(
        inherited_interpretive=inherited_interpretive,
        inherited_structured=inherited_structured_model_dump,
        section_title=section["title"],
        hierarchy_str=hierarchy_str,
        section_text=section["text"],
    )

    with using_attributes(
        metadata={
            "page_id": section["page_id"],
            "section_title": section["title"],
        },
    ):
        for _ in range(3):
            try:
                result = CombinedSectionOutput.model_validate(llm.invoke(prompt))
                break
            except Exception as exc:
                logger.warn("Error during invoking llm: %s. Retrying", exc)

    for g in result.graves:
        g = GraveRecordWithPageIdAndSectionTitle(
            **g.model_dump(),
            page_id=section["page_id"],
            section_title=section["title"],
        )

        logger.info(
            "Extracted grave", grave=g.model_dump(exclude_none=True, exclude_unset=True)
        )
        state.extracted_records.append(g)

    context_manager.push(
        result.shared_context.structured,
        result.shared_context.interpretive,
        section["heading_level"],
    )

    return state


class CSVExportRow(GraveRecord):
    page_id: int


@timed
def execute_agent(
    sections: list[SectionData],
    llm: BaseChatModel,
    grave_extraction_prompt_strategy: PromptStrategy,
) -> pd.DataFrame:
    logger.info("Agent started", model_name=llm.name)

    context_manager = ContextManager()
    state = ExtractionState()

    structured_llm = llm.with_structured_output(CombinedSectionOutput).with_retry(
        stop_after_attempt=10, wait_exponential_jitter=True
    )

    df = pd.DataFrame()

    for section in sections:
        logger.info("Processing new section", title=section["title"])

        # Update hierarchy
        state.push_section(section)

        if section["text"] is None:
            logger.warning("Section has no text, skipping")
            state.clear_buffer()
            continue

        if section["heading_level"] is None:
            raise Exception(
                f"No heading level set for section {section['title']} (p. {section['page_id']})"
            )

        state.section_buffer = section

        logger.info("Extracting records from section")
        before_records = len(state.extracted_records)
        state = extract_from_section(
            structured_llm,  # type: ignore
            state,
            context_manager,
            grave_extraction_prompt_strategy,
        )
        after_records = len(state.extracted_records)
        logger.info("Extracted %d records", after_records - before_records)

        df = pd.DataFrame([r.model_dump() for r in state.extracted_records])
        df.to_csv(
            "outputs/experiments/grave-extraction/temp_raw.csv",
            index=False,
        )

        eval_df = transform_df_for_evaluation(df)
        eval_df.to_csv(
            "outputs/experiments/grave-extraction/temp_eval.csv",
            index=False,
        )

        state.clear_buffer()

        # wait 20 seconds to not run into gwdg rate limits (200 requests per hour)
        time.sleep(20)

    logger.info("Agent finished")
    return df


def transform_df_for_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    # Target schema order
    cols = [
        "page_id",
        "section_title",
        "fundort",
        "bezirk",
        "grab_identifikation",
        "bestattung_geschlecht",
        "bestattung_alter",
        "bestattung_ritus",
        "bestattung_lage",
        "bestattung_orientierung",
        "bestattung_störung",
        "grabeinbauten",
        "grube_form",
        "grube_länge",
        "grube_breite",
        "grube_tiefe",
        "beigaben_liste",
        "referenzierte_abbildungen",
        "Bemerkung",
    ]

    if len(df) == 0:
        return pd.DataFrame(columns=cols)

    _df = df.copy()

    def safe_eval(val: Any) -> Any:
        """
        Parses a string representation of a list/dict back into a Python object.
        Returns the original value if it's already an object, or an empty list if NaN/Error.
        """
        if isinstance(val, (list, dict)):
            return val

        if pd.isna(val) or val == "":
            return []

        if isinstance(val, str):
            try:
                # ast.literal_eval is safer than eval()
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                # Handle cases where the string is malformed
                return []

        return []

    if "bestattungen" in _df.columns:
        _df["bestattungen"] = _df["bestattungen"].apply(safe_eval)

    if "beigaben_liste" in _df.columns:
        _df["beigaben_liste"] = _df["beigaben_liste"].apply(safe_eval)

    # Helper to explode burial dicts into columns
    def parse_burials(row):
        if not isinstance(row, list) or not row:
            return pd.Series(dtype=object)
        mapping = {
            "geschlecht": "bestattung_geschlecht",
            "alter": "bestattung_alter",
            "ritus": "bestattung_ritus",
            "lage": "bestattung_lage",
            "orientierung": "bestattung_orientierung",
            "störung": "bestattung_störung",
        }
        data = {tgt: [d.get(src) for d in row] for src, tgt in mapping.items()}
        return pd.Series({k: tuple(v) if len(v) > 1 else v[0] for k, v in data.items()})

    # Helper to format beigaben_liste
    def parse_funds(row):
        if not isinstance(row, list) or not row:
            return None
        items = [
            f"{i['amount']}x{i['name']}" if i.get("amount", 1) > 1 else i["name"]
            for i in row
        ]
        return ", ".join(items).lower()

    # Process
    burial_df = _df["bestattungen"].apply(parse_burials)
    _df["beigaben_liste"] = _df["beigaben_liste"].apply(parse_funds)

    # Merge, reindex to schema (creates missing cols like 'Bemerkung'), and return
    return pd.concat([_df, burial_df], axis=1).reindex(columns=cols)
