import json
from unittest.mock import patch

import pandas as pd
import pytest
from grave_extraction.evaluation.grave_metrics import (
    _calculate_beigaben_f1,
    _parse_and_expand_beigaben,
    compare_graves,
)


def run_comparison(
    df_gt, df_pred, tmp_path, monkeypatch=None, inputs=None, mock_read_csv=False
):
    """
    Helper to save DFs to temp CSVs, setup inputs/mocks, and run compare_graves.
    """
    gt_path = tmp_path / "gt.csv"
    pred_path = tmp_path / "pred.csv"
    out_csv = tmp_path / "out.csv"
    out_json = tmp_path / "out.json"

    # Save CSVs
    df_gt.to_csv(gt_path, index=False)
    df_pred.to_csv(pred_path, index=False)

    # Setup User Input Mock
    if monkeypatch and inputs:
        input_iter = iter(inputs)
        monkeypatch.setattr("builtins.input", lambda _: next(input_iter))

    # Setup read_csv Mock (to preserve Tuples for specific tests)
    if mock_read_csv and monkeypatch:

        def custom_read_csv(filepath, *args, **kwargs):
            if str(filepath) == str(gt_path):
                return df_gt
            if str(filepath) == str(pred_path):
                return df_pred
            return pd.read_csv(filepath, *args, **kwargs)

        monkeypatch.setattr(pd, "read_csv", custom_read_csv)

    # Run the function
    res_df = compare_graves(str(gt_path), str(pred_path), str(out_csv), str(out_json))

    return res_df


def test_compare_strings(caplog, tmp_path):
    cols = ["page_id", "grab_identifikation", "bestattung_geschlecht"]

    df_gt = pd.DataFrame(
        [
            [1, "Grab 1", "m"],
            [1, "Grab 2", "w"],
            [1, "Grab 3", None],
            [1, "Grab 4", "m"],
        ],
        columns=cols,
    )

    df_pred = pd.DataFrame(
        [
            [1, "Grab 1", "m"],
            [1, "Grab 2", "m"],
            [1, "Grab 3", None],
            [1, "Grab 4", None],
        ],
        columns=cols,
    )

    res = run_comparison(df_gt, df_pred, tmp_path)

    assert res.loc[0, "bestattung_geschlecht"] == 1
    assert res.loc[1, "bestattung_geschlecht"] == 0
    assert res.loc[2, "bestattung_geschlecht"] == 1
    assert res.loc[3, "bestattung_geschlecht"] == 0


def test_compare_numerics(caplog, tmp_path):
    cols = ["page_id", "grab_identifikation", "grube_länge"]
    df_gt = pd.DataFrame([[1, "G1", 100.0], [1, "G2", 100.0]], columns=cols)
    df_pred = pd.DataFrame([[1, "G1", 100.0], [1, "G2", 100.05]], columns=cols)

    res = run_comparison(df_gt, df_pred, tmp_path)

    # 100.0 == 100.0 -> 1
    # 100.0 vs 100.05 (diff 0.05 < 0.1) -> 1
    assert res.loc[0, "grube_länge"] == 1
    assert res.loc[1, "grube_länge"] == 1


def test_manual_columns_interactive_scoring(caplog, monkeypatch, tmp_path):
    # 'fundort' is a manual column
    cols = ["page_id", "grab_identifikation", "fundort"]

    df_gt = pd.DataFrame([[1, "Grab 1", "FA"]], columns=cols)
    df_pred = pd.DataFrame([[1, "Grab 1", "FA_"]], columns=cols)

    # Prompt user for score: 1 = Correct
    res = run_comparison(df_gt, df_pred, tmp_path, monkeypatch, inputs=["1"])

    # Result should be the score
    assert res.loc[0, "fundort"] == 1


def test_multiple_bestattungen(caplog, tmp_path, monkeypatch):
    """
    Verifies tuple handling.
    """
    cols = ["page_id", "grab_identifikation", "bestattung_geschlecht"]

    df_gt = pd.DataFrame([[1, "Grab 1", "m"]], columns=cols)
    # Simulate a tuple coming from the prediction (mocking read_csv behavior)
    df_pred = pd.DataFrame([[1, "Grab 1", ("m", "w")]], columns=cols)

    res = run_comparison(df_gt, df_pred, tmp_path, monkeypatch, mock_read_csv=True)

    # Check Value: Should be 0 (Penalty), not the tuple
    val = res.loc[0, "bestattung_geschlecht"]
    assert val == 0, f"Expected score 0 for tuple penalty, got {val}"


def test_compare_graves_full_integration(caplog, monkeypatch, tmp_path):
    cols = ["page_id", "grab_identifikation", "grube_länge", "fundort"]

    df_gt = pd.DataFrame(
        [
            [1, "G1", 100.0, "LocA"],
            [1, "G2", 100.0, "LocA"],
        ],
        columns=cols,
    )

    df_pred = pd.DataFrame(
        [
            [1, "G1", 100.0, "LocA"],
            [1, "G2", 200.0, "LocB"],
        ],
        columns=cols,
    )

    # G1: fundort match (auto 1 via string check inside _get_manual_score) -> Prompt Skipped?
    # Actually _get_manual_score attempts auto-match first. "LocA"=="LocA" -> 1.
    # G2: fundort mismatch ("LocA" vs "LocB") -> Prompt -> Input "0"

    inputs = ["0"]
    run_comparison(df_gt, df_pred, tmp_path, monkeypatch, inputs=inputs)

    # Check JSON
    with open(tmp_path / "out.json") as f:
        d = json.load(f)

    # grube_länge: 1 correct (G1), 1 incorrect (G2, 200 vs 100) -> 50%
    assert d["metrics"]["per_column"]["grube_länge"]["score_sum"] == 1.0
    # fundort: 1 correct (G1), 1 incorrect (G2 user input 0) -> 50%
    assert d["metrics"]["per_column"]["fundort"]["score_sum"] == 1.0


def test_accuracy_ignores_skipped_rows(caplog, monkeypatch, tmp_path):
    """
    Verifies that rows resulting in None (NaN) in the result DataFrame
    (e.g., skipped comparisons) are NOT counted in the accuracy denominator.
    Also verifies the new JSON key 'score_sum' is used.
    """
    cols = ["page_id", "grab_identifikation", "grube_länge"]

    # GT has 3 entries
    df_gt = pd.DataFrame(
        [
            [1, "Grab 1", 100.0],  # Match -> Correct
            [1, "Grab 2", 100.0],  # Match -> Incorrect
            [1, "Grab 3", 100.0],  # No Match -> User Skips -> Result should be NaN
        ],
        columns=cols,
    )

    # Pred only has 2 entries (Grab 3 is missing)
    df_pred = pd.DataFrame(
        [
            [1, "Grab 1", 100.0],  # Correct value
            [1, "Grab 2", 200.0],  # Incorrect value
        ],
        columns=cols,
    )

    # Input "0" to simulate user selecting "NO MATCH (Skip)" for Grab 3
    res = run_comparison(df_gt, df_pred, tmp_path, monkeypatch, inputs=["0"])

    # Verify DataFrame contents
    assert res.loc[0, "grube_länge"] == 1
    assert res.loc[1, "grube_länge"] == 0
    assert pd.isna(res.loc[2, "grube_länge"])

    # Verify JSON Metrics
    out_json_path = tmp_path / "out.json"

    with open(out_json_path, "r") as f:
        data = json.load(f)

    stats = data["metrics"]["per_column"]["grube_länge"]

    # Calculation should be:
    # Total Valid = 2 (Row 0 and Row 1)
    # Score Sum = 1.0 (Row 0 is 1.0, Row 1 is 0.0)
    # Accuracy = 1 / 2 = 0.5

    assert stats["total"] == 2
    assert stats["score_sum"] == 1.0
    assert stats["metric_score"] == 0.5


@pytest.mark.parametrize(
    "input_str, expected_list",
    [
        ("2x Perle, 1x Messer", ["Perle", "Perle", "Messer"]),
        ("2xperle, 3xbead", ["perle", "perle", "bead", "bead", "bead"]),
        ("2x perle (bronze), gefaess", ["perle (bronze)", "perle (bronze)", "gefaess"]),
        ("messer, perle, fibel", ["messer", "perle", "fibel"]),
        ("", []),
        ("   ,   ", []),
    ],
)
def test_parse_beigaben_variations(input_str, expected_list):
    """Test various string formats and counts."""
    assert _parse_and_expand_beigaben(input_str) == expected_list


def test_f1_perfect_match():
    """
    Scenario: GT matches Pred exactly (2 items).
    User selects '1' (Match) for the first item.
    User selects '1' (Match) for the second item.
    """
    gt = "2xA"
    pred = "2xA"

    with patch("builtins.input", side_effect=["1", "1"]):
        p, r, f1 = _calculate_beigaben_f1(gt, pred)

    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0


def test_f1_partial_match():
    """
    Scenario: GT=['A', 'B'], Pred=['A', 'C']
    1. User matches A -> A (Input '1')
    2. User cannot match B -> C (Input '0')
    Result: TP=1, FN=1, FP=1 (C remains)
    """
    gt = "A, B"
    pred = "A, C"

    with patch("builtins.input", side_effect=["1", "0"]):
        p, r, f1 = _calculate_beigaben_f1(gt, pred)

    assert p == 0.5
    assert r == 0.5
    assert f1 == 0.5


def test_hallucination_mixed_columns(tmp_path, monkeypatch):
    """
    Verifies hallucinations (GT=Empty, Pred=Value) are recorded for
    both automatic (numeric) and manual columns as a list of indices.
    """
    # 'grube_länge' is in COLS_AUTO_NUM
    # 'fundort' is in COLS_MANUAL
    cols = ["page_id", "grab_identifikation", "grube_länge", "fundort"]

    # GT Data: All target values are None/NaN
    df_gt = pd.DataFrame(
        [
            [1, "Grab 1", None, None],  # Row 0
            [1, "Grab 2", None, None],  # Row 1
            [1, "Grab 3", None, None],  # Row 2
        ],
        columns=cols,
    )

    # Pred Data:
    # Row 0: Hallucination in 'grube_länge' AND 'fundort'
    # Row 1: No Hallucination (Matches NaN)
    # Row 2: Hallucination in 'grube_länge' only
    df_pred = pd.DataFrame(
        [
            [1, "Grab 1", 100, "Wien"],
            [1, "Grab 2", None, None],
            [1, "Grab 3", 200, None],
        ],
        columns=cols,
    )

    # Mock Input:
    # For Row 0, 'fundort' (Manual): GT=None, Pred="Wien".
    # System will prompt "Correct? [1/0]". We input "0" (Incorrect) to flag as hallucination.
    monkeypatch.setattr("builtins.input", lambda _: "0")

    # Run comparison
    run_comparison(df_gt, df_pred, tmp_path)

    # Load JSON output
    out_json_path = tmp_path / "out.json"
    with open(out_json_path, "r") as f:
        data = json.load(f)

    h_stats = data["statistics"]["hallucinations_per_column"]

    # Check 'grube_länge' (Auto Numeric)
    # Should detect hallucinations at index 0 and 2
    assert "grube_länge" in h_stats
    assert sorted(h_stats["grube_länge"]) == [0, 2]

    # Check 'fundort' (Manual)
    # Should detect hallucination at index 0 only
    assert "fundort" in h_stats
    assert h_stats["fundort"] == [0]


def test_f1_hallucination_only(tmp_path):
    """
    Verifies that if GT is empty and Pred has items for beigaben_liste,
    it is correctly recorded as a hallucination in the JSON summary.
    """
    cols = ["page_id", "grab_identifikation", "beigaben_liste"]

    # Scenario: GT is empty (NaN or empty string), Pred has "Messer"
    # This is row index 0
    df_gt = pd.DataFrame([[1, "Grab 1", None]], columns=cols)
    df_pred = pd.DataFrame([[1, "Grab 1", "Messer"]], columns=cols)

    # Run comparison (No user input needed for hallucinations, returns 0.0 auto)
    run_comparison(df_gt, df_pred, tmp_path)

    # Load the generated JSON
    out_json_path = tmp_path / "out.json"
    with open(out_json_path, "r") as f:
        data = json.load(f)

    # Assert hallucination was counted
    h_stats = data["statistics"]["hallucinations_per_column"]

    assert "beigaben_liste" in h_stats
    # Expect a list containing the index of the row where the hallucination occurred (0)
    assert h_stats["beigaben_liste"] == [0]


def test_f1_hallucination_only_2():
    """
    Scenario: GT is empty, Pred has items.
    Should return 0.0 immediately without asking for input.
    """
    gt = None
    pred = "Messer"

    with patch("builtins.input") as mock_input:
        p, r, f1 = _calculate_beigaben_f1(gt, pred)

    assert (p, r, f1) == (0.0, 0.0, 0.0)
    mock_input.assert_not_called()


def test_f1_over_prediction():
    """
    Scenario: Pred has more counts than GT.
    GT: "A"
    Pred: "2xA" (Expands to ['A', 'A'])
    User matches the first A. The second A is leftover FP.
    """
    gt = "A"
    pred = "2xA"

    # User selects '1' (match first available A)
    with patch("builtins.input", side_effect=["1"]):
        p, r, f1 = _calculate_beigaben_f1(gt, pred)

    # TP=1, FN=0, FP=1
    # Prec=0.5, Rec=1.0
    assert p == 0.5
    assert r == 1.0
    assert f1 == pytest.approx(0.666666, abs=1e-5)


def test_f1_missed_prediction():
    """
    Scenario: Pred is empty, GT has items.
    Should return 0.0 immediately.
    """
    gt = "Perle"
    pred = ""

    with patch("builtins.input") as mock_input:
        p, r, f1 = _calculate_beigaben_f1(gt, pred)

    assert (p, r, f1) == (0.0, 0.0, 0.0)
    mock_input.assert_not_called()


def test_compare_graves_integration_with_beigaben(caplog, monkeypatch, tmp_path):
    """
    End-to-End test to ensure 'beigaben_liste' triggers the F1 logic
    and stores the float result in the output CSV.
    """
    cols = ["page_id", "grab_identifikation", "beigaben_liste"]

    # GT: Two Perles
    df_gt = pd.DataFrame([[1, "Grab 1", "2x Perle"]], columns=cols)

    # Pred: One Perle
    df_pred = pd.DataFrame([[1, "Grab 1", "1x Perle"]], columns=cols)

    # Expected Logic:
    # 1. Expand GT: [Perle, Perle]
    # 2. Expand Pred: [Perle]
    # 3. Prompt for 1st GT item: User inputs '1' (Match)
    # 4. Prompt for 2nd GT item: Auto-Missed because Pred list is empty.
    # TP=1, FN=1, FP=0.
    # Prec=1.0, Rec=0.5, F1=0.6666...

    inputs = ["1"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))

    res = run_comparison(df_gt, df_pred, tmp_path)

    f1_result = res.loc[0, "beigaben_liste"]
    assert isinstance(f1_result, float)
    assert f1_result == pytest.approx(0.666666, abs=1e-5)


def test_gt_tuple_valid_matches(caplog, tmp_path, monkeypatch):
    """
    Tests that if Ground Truth contains a tuple (e.g., (100, 105) or ('m', 'w')),
    a prediction matching ANY of those values counts as correct (1).
    """
    cols = ["page_id", "grab_identifikation", "grube_länge", "bestattung_geschlecht"]

    # GT has Tuples
    df_gt = pd.DataFrame(
        [
            [1, "G1", (100.0, 110.0), ("m", "w")],  # Numeric Tuple, Cat Tuple
            [1, "G2", (50.0, 60.0), "m"],  # Numeric Tuple, Single Cat
            [1, "G3", (10.0, 20.0), ("x", "y")],  # No matches
        ],
        columns=cols,
    )

    # Pred has Single values
    df_pred = pd.DataFrame(
        [
            [1, "G1", 110.0, "w"],  # Matches 2nd item in Num Tuple, 2nd in Cat Tuple
            [1, "G2", 55.0, "m"],  # No match in Num Tuple (50 or 60), Match in Cat
            [1, "G3", 99.0, "z"],  # No matches anywhere
        ],
        columns=cols,
    )

    # Use mock_read_csv to ensure Tuples are passed as Python objects to the logic
    res = run_comparison(df_gt, df_pred, tmp_path, monkeypatch, mock_read_csv=True)

    # Row 0: G1 (Both Valid)
    # 110.0 is in (100.0, 110.0) -> Correct
    assert res.loc[0, "grube_länge"] == 1
    # 'w' is in ('m', 'w') -> Correct
    assert res.loc[0, "bestattung_geschlecht"] == 1

    # Row 1: G2 (Mixed)
    # 55.0 is NOT in (50.0, 60.0) -> Incorrect
    assert res.loc[1, "grube_länge"] == 0
    # 'm' == 'm' -> Correct
    assert res.loc[1, "bestattung_geschlecht"] == 1

    # Row 2: G3 (Both Invalid)
    assert res.loc[2, "grube_länge"] == 0
    assert res.loc[2, "bestattung_geschlecht"] == 0


def test_f1_bulk_auto_count_list(monkeypatch):
    gt = " 10x Perle, 1xdraht"
    # Pool contains 5 'prle', 5 'bead', 1 'drahtring'
    pred = "5x prle, 5xplre, 1xdrahtring"

    inputs = [
        "prle, typo_string",  # 1. Invalid input (typo_string not in pool) -> Logic should retry
        "prle, plre",  # 2. Valid list -> System auto-counts 5+5=10.
        "1",  # 3. Match 'draht' -> 'drahtring' (Single item logic)
    ]

    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))

    p, r, f1 = _calculate_beigaben_f1(gt, pred)

    # Logic:
    # GT "Perle" (10) matches 5 "prle" + 5 "bead" = 10. TP += 10. FN += 0.
    # GT "Draht" (1) matches "drahtring" (1). TP += 1.
    # Remaining Pool: Empty. FP = 0.

    expected_precision = 1.0
    expected_recall = 1.0
    expected_f1 = 1.0

    assert p == pytest.approx(expected_precision)
    assert r == pytest.approx(expected_recall)
    assert f1 == pytest.approx(expected_f1)


def test_matches_field_correctness_with_global_indices(tmp_path):
    """
    Verifies that the 'matches' field in the output JSON contains the correct
    [gt_index, pred_index] pairs, specifically preserving the GLOBAL index
    of the prediction DataFrame, not the relative index within a page subset.
    """
    cols = ["page_id", "grab_identifikation", "grube_länge"]

    # Ground Truth
    # Idx 0: Page 10, G1
    # Idx 1: Page 20, G2
    df_gt = pd.DataFrame(
        [
            [10, "G1", 100],
            [20, "G2", 200],
        ],
        columns=cols,
    )

    # Predictions
    # Idx 0: Page 5,  Noise (Global 0)
    # Idx 1: Page 10, G1    (Global 1) -> Match for GT Idx 0
    # Idx 2: Page 10, Noise (Global 2)
    # Idx 3: Page 20, G_Miss(Global 3)
    # Idx 4: Page 20, G2    (Global 4) -> Match for GT Idx 1
    df_pred = pd.DataFrame(
        [
            [5, "Noise", 999],
            [10, "G1", 100],
            [10, "Noise", 999],
            [20, "Miss", 999],
            [20, "G2", 200],
        ],
        columns=cols,
    )

    # Run comparison
    run_comparison(df_gt, df_pred, tmp_path)

    # Check JSON
    out_json = tmp_path / "out.json"
    with open(out_json, "r") as f:
        data = json.load(f)

    assert "matches" in data, "JSON output missing 'matches' field"
    matches = data["matches"]

    # Sort matches by GT index to ensure deterministic assertion
    matches.sort(key=lambda x: x[0])

    # Expectation:
    # GT 0 ("G1") matches Pred 1 ("G1") -> [0, 1]
    # GT 1 ("G2") matches Pred 4 ("G2") -> [1, 4]

    # Note: If the logic incorrectly used relative indices (subset indices):
    # - Page 10 subset would be [Pred 1, Pred 2]. "G1" is at index 0 of this subset.
    # - Page 20 subset would be [Pred 3, Pred 4]. "G2" is at index 1 of this subset.
    # Incorrect result would be [[0, 0], [1, 1]].

    assert matches == [[0, 1], [1, 4]]


def test_matches_field_manual_selection_preserves_global_index(tmp_path, monkeypatch):
    """
    Verifies that when a user manually resolves an ambiguous match (no exact string match),
    the system correctly records the Global Index of the selected prediction row
    in the 'matches' JSON field.
    """
    cols = ["page_id", "grab_identifikation", "grube_länge"]

    # GT: One entry looking for "Grave Target" on Page 1
    df_gt = pd.DataFrame([[1, "Grave Target", 100]], columns=cols)

    # Pred:
    # Idx 0: Page 9 (Noise)
    # Idx 1: Page 1 "Candidate A" (First option on page)
    # Idx 2: Page 1 "Candidate B" (Second option on page - The one we want)
    df_pred = pd.DataFrame(
        [
            [9, "Noise", 0],
            [1, "Candidate A", 50],
            [1, "Candidate B", 100],
        ],
        columns=cols,
    )

    # Logic flow during execution:
    # 1. System looks for "Grave Target" on Page 1.
    # 2. Finds subset rows [1, 2] (Global Indices).
    # 3. No exact text match found ("Candidate A" != "Grave Target").
    # 4. Enters Manual Selection Mode.
    # 5. Displays candidates:
    #    [1] Candidate A (Global Idx 1)
    #    [2] Candidate B (Global Idx 2)
    # 6. We mock input "2" to select the second option.

    inputs = ["2"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))

    run_comparison(df_gt, df_pred, tmp_path)

    # Verify JSON Output
    with open(tmp_path / "out.json") as f:
        data = json.load(f)

    # Expected Match: [GT Index 0, Pred Global Index 2]
    # If it wrongly recorded the subset index, it would be [0, 1].
    assert data["matches"] == [[0, 2]]


def test_manual_ambiguous_multiple_match_selection(tmp_path, monkeypatch):
    """
    Verifies that selecting the 'MULTIPLE MATCHES' option in the manual resolution menu
    correctly treats the entry as a duplicate:
    1. Sets output row to NaN.
    2. Adds to duplicates statistics in JSON.
    3. Does not add to valid matches list.
    """
    cols = ["page_id", "grab_identifikation", "grube_länge"]

    # GT: One entry looking for "G1" on Page 1
    df_gt = pd.DataFrame([[1, "G1", 100]], columns=cols)

    # Pred: Two similar but not identical candidates on Page 1
    # This forces the manual ambiguity menu to appear because no exact string match exists.
    df_pred = pd.DataFrame(
        [
            [1, "G1 candidate_a", 100],
            [1, "G1 candidate_b", 100],
        ],
        columns=cols,
    )

    # Logic:
    # 1. System finds 2 candidates.
    # 2. 'opt_multiple' is calculated as len(candidates) + 1 = 3.
    # 3. Menu Options: [1] Cand A, [2] Cand B, [0] None, [3] MULTIPLE MATCHES.
    # 4. We input "3".

    inputs = ["3"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))

    res_df = run_comparison(df_gt, df_pred, tmp_path)

    # Assertion 1: Verify DataFrame row is NaN (skipped/duplicate behavior)
    # _set_row_null sets numeric columns to NaN.
    assert pd.isna(res_df.loc[0, "grube_länge"]), (
        "Result row should be null for duplicate selection"
    )

    # Assertion 2: Verify JSON Statistics
    with open(tmp_path / "out.json") as f:
        data = json.load(f)

    # Should be recorded in duplicate_matches
    duplicates = data["statistics"]["duplicate_matches"]
    assert duplicates["count"] == 1
    assert duplicates["items"][0] == 0

    # Should NOT be in the 'matches' list (matches are only for valid pairs)
    assert len(data["matches"]) == 0


def test_tuple_error_indices_json_output(tmp_path):
    """
    Verifies that when predictions contain ambiguous tuples (lists/tuples in string format),
    the system correctly records:
    1. The count of errors.
    2. The specific Ground Truth Indices where these errors occurred in the JSON output.
    """
    cols = ["page_id", "grab_identifikation", "grube_länge"]

    # Ground Truth: 3 entries
    # Idx 0: Target for tuple error
    # Idx 1: Control (Normal prediction)
    # Idx 2: Target for tuple error
    df_gt = pd.DataFrame(
        [
            [1, "G1", 100],
            [1, "G2", 200],
            [1, "G3", 300],
        ],
        columns=cols,
    )

    # Predictions
    # Idx 0: "G1" has a tuple string "(100, 105)" -> Ambiguous -> Tuple Error
    # Idx 1: "G2" has a scalar 200 -> Correct -> No Error
    # Idx 2: "G3" has a tuple string "(300, 310)" -> Ambiguous -> Tuple Error
    df_pred = pd.DataFrame(
        [
            [1, "G1", "(100, 105)"],
            [1, "G2", 200],
            [1, "G3", "(300, 310)"],
        ],
        columns=cols,
    )

    # Run comparison
    # We don't need to mock read_csv here because the source code uses
    # ast.literal_eval() to parse strings starting with "(" into tuples.
    run_comparison(df_gt, df_pred, tmp_path)

    # Load resulting JSON
    out_json = tmp_path / "out.json"
    with open(out_json, "r") as f:
        data = json.load(f)

    # Navigate to tuple_errors
    stats = data["statistics"]["tuple_errors"]

    # Assertions
    assert "grube_länge" in stats, "Column should be listed in tuple_errors"

    err_data = stats["grube_länge"]

    # Check Count
    assert err_data["count"] == 2

    # Check Indices: Should be GT index 0 and GT index 2
    assert err_data["gt_indices"] == [0, 2]


def test_omission_detection(monkeypatch, tmp_path):
    """
    Verifies that if GT has values and Pred is empty (NaN/None),
    it is correctly recorded as an omission in the JSON summary.
    """
    cols = [
        "page_id",
        "grab_identifikation",
        "grube_länge",
        "fundort",
        "beigaben_liste",
    ]

    # GT Data: Values present
    df_gt = pd.DataFrame(
        [
            [1, "Grab 1", 150, "Linz", "Messer"],  # Row 0
            [1, "Grab 2", 180, "Wien", "Perle"],  # Row 1
        ],
        columns=cols,
    )

    # Pred Data:
    # Row 0: Omission in 'grube_länge' and 'beigaben_liste'
    # Row 1: Omission in 'fundort'
    df_pred = pd.DataFrame(
        [
            [1, "Grab 1", None, "Linz", None],  # 'grube_länge'=None, 'beigaben'=None
            [1, "Grab 2", 180, None, "Perle"],  # 'fundort'=None
        ],
        columns=cols,
    )

    # Mock user input for Row 1 'beigaben_liste' matching ("Perle" vs "Perle").
    # Row 0 triggers a silent edge case (GT exists, Pred empty) -> No input needed.
    # Row 1 triggers the interactive matcher. We simply select "1" (the match).
    monkeypatch.setattr("builtins.input", lambda _: "1")

    run_comparison(df_gt, df_pred, tmp_path)

    out_json_path = tmp_path / "out.json"
    with open(out_json_path, "r") as f:
        data = json.load(f)

    o_stats = data["statistics"]["omissions_per_column"]

    # Check Auto Numeric ('grube_länge' missing in Row 0)
    assert "grube_länge" in o_stats
    assert o_stats["grube_länge"] == [0]

    # Check Manual ('fundort' missing in Row 1)
    assert "fundort" in o_stats
    assert o_stats["fundort"] == [1]

    # Check Beigaben ('Messer' missing in Row 0)
    assert "beigaben_liste" in o_stats
    assert o_stats["beigaben_liste"] == [0]


def test_manual_hallucination_logic_conditional_on_match(tmp_path, monkeypatch):
    """
    Verifies that for Manual Columns:
    1. If GT is NaN and Pred is not NaN (technically a potential hallucination).
    2. But the user manually marks it as 'Correct' (1) (e.g. accepting "Nothing" as None).
    3. Then it is NOT recorded as a hallucination.

    Contrast with:
    4. User marks it as 'Incorrect' (0).
    5. It IS recorded as a hallucination.
    """
    cols = ["page_id", "grab_identifikation", "fundort"]

    # Row 0: GT is None, Pred is "Nothing". User will accept this (Input 1).
    # Row 1: GT is None, Pred is "Something". User will reject this (Input 0).
    df_gt = pd.DataFrame(
        [
            [1, "G1", None],
            [1, "G2", None],
        ],
        columns=cols,
    )

    df_pred = pd.DataFrame(
        [
            [1, "G1", "Nothing"],
            [1, "G2", "Something"],
        ],
        columns=cols,
    )

    # Inputs sequence:
    # 1. Prompt for G1 'fundort': Input "1" (Yes/Correct)
    # 2. Prompt for G2 'fundort': Input "0" (No/Incorrect)
    inputs = ["1", "0"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))

    run_comparison(df_gt, df_pred, tmp_path)

    # Check JSON output
    out_json = tmp_path / "out.json"
    with open(out_json, "r") as f:
        data = json.load(f)

    h_stats = data["statistics"]["hallucinations_per_column"]

    # 'fundort' should be listed in hallucinations
    assert "fundort" in h_stats

    # We expect ONLY index 1 (G2) to be in the list.
    # Index 0 (G1) should be excluded because the user marked it as correct.
    assert h_stats["fundort"] == [1]


def test_manual_omission_is_automatic_and_strict(tmp_path, monkeypatch):
    """
    Verifies that for Omissions in manual columns:
    1. GT has value, Pred is NaN.
    2. System does NOT prompt the user.
    3. Automatically records as 0 (Incorrect) and as Omission.
    """
    cols = ["page_id", "grab_identifikation", "fundort"]

    # Row 0: GT "Hamburg", Pred None.
    # Should be auto-detected as Omission.
    df_gt = pd.DataFrame(
        [
            [1, "G1", "Hamburg"],
        ],
        columns=cols,
    )

    df_pred = pd.DataFrame(
        [
            [1, "G1", None],
        ],
        columns=cols,
    )

    # If the system prompts, this will fail because inputs list is empty.
    inputs = []
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))

    run_comparison(df_gt, df_pred, tmp_path)

    out_json = tmp_path / "out.json"
    with open(out_json, "r") as f:
        data = json.load(f)

    o_stats = data["statistics"]["omissions_per_column"]

    assert "fundort" in o_stats
    assert o_stats["fundort"] == [0]


def test_page_id_scope_isolation(tmp_path):
    """
    Ensures that a Grave ID match is strictly scoped to the correct Page ID.
    If 'Grab 1' exists on Page 2 in predictions, it should NOT match 'Grab 1' on Page 1 in GT.
    """
    cols = ["page_id", "grab_identifikation", "grube_länge"]

    # GT looks for Grab 1 on Page 1
    df_gt = pd.DataFrame([[1, "Grab 1", 100]], columns=cols)

    # Pred has Grab 1, but on Page 2
    df_pred = pd.DataFrame([[2, "Grab 1", 100]], columns=cols)

    res = run_comparison(df_gt, df_pred, tmp_path)

    # Should be treated as NO MATCH found (Row becomes NaN/Null)
    # because the subset for Page 1 will be empty.
    assert pd.isna(res.loc[0, "grube_länge"])

    # Verify it is recorded as a 'no_match' in statistics, not a wrong prediction
    with open(tmp_path / "out.json") as f:
        data = json.load(f)

    assert data["statistics"]["no_matches"]["count"] == 1
    data["statistics"]["no_matches"]["items"][0] == 0


def test_categorical_mixed_type_matching(tmp_path):
    """
    Verifies that categorical columns handle type mismatches robustly:
    1. Int (GT) vs String (Pred) -> Should match via normalization.
    2. Int (GT) vs Float (Pred)  -> Should match via numeric equality check in code.
    """
    cols = ["page_id", "grab_identifikation", "phase"]  # 'phase' treated as categorical

    # GT uses Integers
    df_gt = pd.DataFrame([[1, "G1", 1], [1, "G2", 1]], columns=cols)

    # Pred uses String and Float
    df_pred = pd.DataFrame(
        [
            [1, "G1", "1"],  # String "1"
            [1, "G2", 1.0],  # Float 1.0
        ],
        columns=cols,
    )

    res = run_comparison(df_gt, df_pred, tmp_path)

    # Both should be 1 (Correct)
    # If logic is strict on types without fallback, these would fail.
    assert res.loc[0, "phase"] == 1
    assert res.loc[1, "phase"] == 1


def test_beigaben_both_empty_is_perfect_score(tmp_path):
    """
    Verifies that if both GT and Pred are empty/NaN/None for beigaben_liste,
    the score is 1.0 (True Negative).
    """
    cols = ["page_id", "grab_identifikation", "beigaben_liste"]

    # Various forms of 'Empty'
    df_gt = pd.DataFrame(
        [[1, "G1", None], [1, "G2", ""], [1, "G3", "   "]], columns=cols
    )

    df_pred = pd.DataFrame(
        [[1, "G1", None], [1, "G2", None], [1, "G3", ""]], columns=cols
    )

    res = run_comparison(df_gt, df_pred, tmp_path)

    # All should result in F1 score of 1.0
    assert res.loc[0, "beigaben_liste"] == 1.0
    assert res.loc[1, "beigaben_liste"] == 1.0
    assert res.loc[2, "beigaben_liste"] == 1.0


def test_global_accuracy_aggregation(tmp_path, monkeypatch):
    """
    Tests that the global accuracy correctly averages:
    - 1.0 (Perfect Numeric)
    - 0.0 (Wrong Manual)
    - 0.5 (Partial Beigaben F1)
    Expected Average: (1.0 + 0.0 + 0.5) / 3 = 0.5
    """
    cols = [
        "page_id",
        "grab_identifikation",
        "grube_länge",
        "fundort",
        "beigaben_liste",
    ]

    df_gt = pd.DataFrame([[1, "G1", 100, "LocA", "2x A"]], columns=cols)

    # Pred:
    # Length: 100 (Score 1.0)
    # Fundort: LocB (Score 0.0) -> User confirms incorrect
    # Beigaben: 1x A (Score 0.5 -> Precision 1.0, Recall 0.5 => F1 0.6666 is usually result,
    # lets use simple 1 match 1 miss scenario for math check).
    # Actually: GT="A, B", Pred="A". TP=1, FN=1, FP=0. P=1, R=0.5, F1=0.6666.
    # To get exactly 0.5 F1 is hard. Let's rely on the JSON math.

    df_pred = pd.DataFrame([[1, "G1", 100, "LocB", "1x A"]], columns=cols)

    # Input "0" for Fundort mismatch.
    # Input "1" for Beigaben match of 'A'.
    inputs = ["0", "1"]
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))

    run_comparison(df_gt, df_pred, tmp_path)

    with open(tmp_path / "out.json") as f:
        data = json.load(f)

    # 1. grube_länge = 1.0
    # 2. fundort = 0.0
    # 3. beigaben: GT(2), Pred(1). TP=1, FN=1. Prec=1.0, Rec=0.5. F1 = 2*(1*0.5)/(1+0.5) = 1/1.5 = 0.6666

    expected_sum = 1.0 + 0.0 + (2 / 3)  # ~1.6666
    expected_count = 3
    expected_avg = expected_sum / expected_count  # ~0.5555

    json_global = data["metrics"]["global_performance"]

    assert json_global == pytest.approx(expected_avg, abs=1e-4)
