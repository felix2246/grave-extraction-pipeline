import pandas as pd
from grave_extraction.extraction.agent import transform_df_for_evaluation


def test_transform_to_ground_truth():
    df_input = pd.DataFrame(
        [
            {
                "page_id": 1,
                "fundort": "A",
                "bestattungen": [{"geschlecht": "m", "ritus": "body"}],
                "beigaben_liste": [{"name": "Ring", "amount": 1}],
            },
            {
                "page_id": 2,
                "fundort": "B",
                "bestattungen": [
                    {"geschlecht": "m", "ritus": "fire"},
                    {"geschlecht": "w", "ritus": "body"},
                ],
                "beigaben_liste": [
                    {"name": "Perle", "amount": 3},
                    {"name": "Dolch", "amount": 1},
                ],
            },
            {"page_id": 3, "fundort": "C", "bestattungen": [], "beigaben_liste": []},
        ]
    )

    res = transform_df_for_evaluation(df_input)

    # Check Schema (including the specific typo 'besattung_ritus')
    assert "bestattung_ritus" in res.columns
    assert "Bemerkung" in res.columns

    # Normal Case: Single values, lowercase list
    assert res.iloc[0]["bestattung_geschlecht"] == "m"
    assert res.iloc[0]["beigaben_liste"] == "ring"

    # Multiple Case: Tuples for burials, 'Nx' format for items
    assert res.iloc[1]["bestattung_geschlecht"] == ("m", "w")
    assert res.iloc[1]["bestattung_ritus"] == ("fire", "body")
    assert res.iloc[1]["beigaben_liste"] == "3xperle, dolch"

    # Empty Case: NaNs and None
    assert pd.isna(res.iloc[2]["bestattung_geschlecht"])
    assert not res.iloc[2]["beigaben_liste"]  # Should be empty string or None
