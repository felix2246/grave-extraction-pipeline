from typing import Optional

from pydantic import BaseModel

from grave_extraction.extraction.context_manager import ContextManager


# Minimal grave record
class GraveRecord(BaseModel):
    fundort: Optional[str] = None
    bezirk: Optional[str] = None


def test_push_and_merge_hierarchical_context():
    manager = ContextManager()

    manager.push(GraveRecord(fundort="Abrahám", bezirk="1900s"), level=1)
    assert manager.merged_structured().fundort == "Abrahám"
    assert manager.merged_structured().bezirk == "1900s"

    manager.push(GraveRecord(fundort="Abrahám I", bezirk="1940"), level=2)
    merged = manager.merged_structured()
    assert merged.fundort == "Abrahám I"  # Overridden value
    assert merged.bezirk == "1940"  # Inherited value


def test_pop_to_level_for_sibling_node():
    """
    Tests a critical case: moving between sibling sections (e.g., Grab 1 -> Grab 2).
    The context from the first sibling must be popped before processing the second.
    """
    manager = ContextManager()
    manager.push(GraveRecord(fundort="Abrahám I"), level=2)

    manager.push(GraveRecord(bezirk="1946"), level=4)
    assert manager.merged_structured().bezirk == "1946"
    assert len(manager.stack) == 2

    # The context from "Grab 1" (level 4) should be removed.
    manager.pop_to_level(current_level=4)

    assert len(manager.stack) == 1
    assert manager.stack[-1].level == 2  # The level 4 context is gone
    assert (
        manager.merged_structured().bezirk is None
    )  # The specific bezirk is no longer in context


def test_pop_to_level_moving_up_hierarchy():
    """
    Tests moving from a deep level back to a higher one (e.g., from a Grab back to a main chapter).
    Ensures all non-ancestor contexts are cleared.
    """
    manager = ContextManager()
    manager.push(GraveRecord(fundort="Site"), level=1)
    manager.push(GraveRecord(bezirk="1950"), level=2)
    manager.push(GraveRecord(), level=4)  # Simulate being deep in the hierarchy
    assert len(manager.stack) == 3

    # This should pop the level 4 and the previous level 2 contexts.
    manager.pop_to_level(current_level=2)

    assert len(manager.stack) == 1
    assert manager.stack[0].level == 1  # Only the true ancestor (level 1) remains
    assert manager.merged_structured().fundort == "Site"
    assert manager.merged_structured().bezirk is None  # Context from level 2 is gone


def test_interpretive_notes_concatenation_and_clearing():
    """
    Tests that interpretive notes are correctly concatenated and cleared
    in sync with the hierarchical traversal.
    """
    manager = ContextManager()
    assert manager.merged_interpretive() is None

    manager.push(GraveRecord(), interpretive=["Site level info."], level=1)
    manager.push(GraveRecord(), interpretive=["Subsection A info."], level=2)
    assert "Site level info." in manager.merged_interpretive()
    assert "Subsection A info." in manager.merged_interpretive()

    # Move to a sibling of the level 2 section. This requires popping first.
    manager.pop_to_level(current_level=2)

    # the note from the previous level 2 section should be gone gone
    assert manager.merged_interpretive() == "Site level info."

    # Add the new sibling's context
    manager.push(GraveRecord(), interpretive=["Subsection B info."], level=2)

    # the notes should be correctly concatenated with the new context
    assert "Site level info." in manager.merged_interpretive()
    assert "Subsection B info." in manager.merged_interpretive()

    # Move all the way up the hierarchy
    manager.pop_to_level(current_level=1)

    # all interpretive notes should be cleared
    assert manager.merged_interpretive() is None
