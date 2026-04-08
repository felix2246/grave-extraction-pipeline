from grave_extraction.extraction.agent import ExtractionState
from grave_extraction.models import SectionData


def test_initial_state():
    """Tests that the initial section_path is empty."""
    state = ExtractionState()

    assert state.section_path == []


def test_descending_hierarchy():
    """
    Tests correct addition of sections when descending the hierarchy.
    (Level 1 -> Level 2 -> Level 3)
    """
    state = ExtractionState()

    state.push_section(
        SectionData(id=0, title="Überschrift", heading_level=0, page_id=0)
    )
    assert state.section_path == ["Überschrift"]

    state.push_section(
        SectionData(id=1, title="Fundplatz Musterhausen", heading_level=1, page_id=0)
    )
    assert state.section_path == ["Überschrift", "Fundplatz Musterhausen"]

    state.push_section(
        SectionData(id=2, title="Gräberfeld A", heading_level=2, page_id=0)
    )
    assert state.section_path == [
        "Überschrift",
        "Fundplatz Musterhausen",
        "Gräberfeld A",
    ]

    state.push_section(SectionData(id=3, title="Grab 1", heading_level=3, page_id=0))
    assert state.section_path == [
        "Überschrift",
        "Fundplatz Musterhausen",
        "Gräberfeld A",
        "Grab 1",
    ]


def test_moving_to_sibling_node_replaces_correctly():
    """
    Simulates switching from 'Grab 1' to 'Grab 2'.
    'Grab 2' must replace 'Grab 1', not be appended.
    """
    state = ExtractionState()

    state.push_section(
        SectionData(id=0, title="Fundplatz Musterhausen", heading_level=1, page_id=0)
    )
    state.push_section(
        SectionData(id=1, title="Gräberfeld A", heading_level=2, page_id=0)
    )
    state.push_section(SectionData(id=2, title="Grab 1", heading_level=3, page_id=0))

    assert state.section_path == ["Fundplatz Musterhausen", "Gräberfeld A", "Grab 1"]

    state.push_section(SectionData(id=3, title="Grab 2", heading_level=3, page_id=0))

    expected_path = ["Fundplatz Musterhausen", "Gräberfeld A", "Grab 2"]
    assert state.section_path == expected_path


def test_ascending_hierarchy():
    """
    Tests correct truncation of the path when ascending the hierarchy.
    (From level 3 back to a new element at level 2)
    """
    state = ExtractionState()

    state.push_section(
        SectionData(id=0, title="Fundplatz Musterhausen", heading_level=1, page_id=0)
    )
    state.push_section(
        SectionData(id=1, title="Gräberfeld A", heading_level=2, page_id=0)
    )
    state.push_section(SectionData(id=2, title="Grab 1", heading_level=3, page_id=0))

    state.push_section(
        SectionData(id=3, title="Gräberfeld B", heading_level=2, page_id=0)
    )

    expected_path = ["Fundplatz Musterhausen", "Gräberfeld B"]
    assert state.section_path == expected_path


def test_jumping_up_multiple_levels():
    """
    Tests a big jump up in the hierarchy.
    (From level 3 back to a new element at level 1)
    """
    state = ExtractionState()

    state.push_section(
        SectionData(id=0, title="Fundplatz Musterhausen", heading_level=1, page_id=0)
    )
    state.push_section(
        SectionData(id=1, title="Gräberfeld A", heading_level=2, page_id=0)
    )
    state.push_section(SectionData(id=2, title="Grab 1", heading_level=3, page_id=0))

    state.push_section(
        SectionData(id=1, title="Fundplatz Neudorf", heading_level=1, page_id=0)
    )

    expected_path = ["Fundplatz Neudorf"]
    assert state.section_path == expected_path
