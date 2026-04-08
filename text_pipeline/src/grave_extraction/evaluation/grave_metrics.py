import ast
import json
import re
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from grave_extraction.logger import logger
from grave_extraction.utils import normalize_text


def _ui_print_grave_header(grave_id: str, page_id: str, idx: int) -> None:
    print("\n" + "=" * 60)
    print(f" GRAVE: {grave_id:<30} | Page: {page_id} | GT-Idx: {idx}")
    print("-" * 60)


def _ui_get_int(max_opt: int, prompt: str) -> int:
    while True:
        try:
            val_str = input(f"   >> {prompt} [0-{max_opt}]: ").strip()
            val = int(val_str)
            if 0 <= val <= max_opt:
                return val
        except ValueError:
            pass


def _ui_get_bool(prompt: str) -> int:
    while True:
        val_str = input(f"   >> {prompt} [1=Yes/0=No]: ").strip()
        if val_str == "1":
            return 1
        if val_str == "0":
            return 0


def _parse_beigaben_structured(text: str) -> list[tuple[int, str]]:
    """
    Parses a string like "2xperle (bronze), 1 messer" into a list of tuples:
    -> [(2, "perle (bronze)"), (1, "messer")]
    """
    if not isinstance(text, str) or not text.strip():
        return []

    raw_items = [x.strip() for x in text.split(",") if x.strip()]
    structured = []

    # Regex: Capture leading numbers, handle optional 'x', capture item name
    pattern = re.compile(r"^\s*(\d+)\s*(?:[xX])?\s*(.+)$")

    for item in raw_items:
        match = pattern.match(item)
        if match:
            count_str = match.group(1)
            name = match.group(2).strip()
            try:
                count = int(count_str)
                if count < 1:
                    count = 1
                structured.append((count, name if name else item))
            except ValueError:
                structured.append((1, item))
        else:
            structured.append((1, item))

    return structured


def _parse_and_expand_beigaben(text: str) -> list[str]:
    """
    Parses a string like "2xperle (bronze), 1 messer" into a flat list:
    -> ["perle (bronze)", "perle (bronze)", "messer"]
    """
    structured = _parse_beigaben_structured(text)
    flat_list = []
    for count, name in structured:
        flat_list.extend([name] * count)
    return flat_list


def _remove_n_matching_predictions(
    preds: list[str], target_name: str, n_remove: int
) -> list[str]:
    """
    Removes up to n_remove items from preds that look like target_name.
    Used for bulk processing to clean up the False Positive pool.
    """
    if n_remove <= 0:
        return preds

    remaining = []
    removed_count = 0
    target_norm = normalize_text(target_name)

    # First pass: Look for partial matches to remove
    for item in preds:
        if removed_count < n_remove and (
            target_norm in normalize_text(item) or normalize_text(item) in target_norm
        ):
            removed_count += 1
        else:
            remaining.append(item)

    # If we didn't remove enough (user said they found 10, but we only found 5 strings matching),
    # we stop here. The user verified the count, so we trust the TP logic.
    # The remaining items in the list might be unrelated FPs.

    return remaining


def _calculate_beigaben_f1(gt_raw: str, pred_raw: str) -> tuple[float, float, float]:
    """
    Interactive F1 calculation with bulk handling for high counts (>= 10).
    User inputs a list of matching strings, system counts and removes them automatically.
    """
    # Parse GT structured (to keep counts) and Pred flat (to verify availability)
    gt_structured = _parse_beigaben_structured(gt_raw)
    pred_flat = _parse_and_expand_beigaben(pred_raw)

    # Silent Edge Cases
    if not gt_structured and not pred_flat:
        return 1.0, 1.0, 1.0
    if not gt_structured and pred_flat:
        return 0.0, 0.0, 0.0
    if gt_structured and not pred_flat:
        return 0.0, 0.0, 0.0

    # Active UI
    print("\n   [?] LIST MATCHING (beigaben_liste)")
    print(f"       GT Raw   : {gt_raw}")
    print(f"       Pred Raw : {pred_raw}")
    print("       " + "." * 40)

    tp = 0
    fn = 0
    available_preds = pred_flat.copy()

    for count, gt_name in gt_structured:
        # High Volume Item (>= 10)
        if count >= 10:
            print(f"\n       BULK ITEM CHECK: '{gt_name}' (GT Count: {count})")
            print(
                f"       Current Unmatched Predictions: {len(available_preds)} items left"
            )

            # Show a preview
            preview_str = ", ".join(set[str](available_preds))
            if len(available_preds) > 25:
                preview_str += "..."
            print(f"       Pred Pool: [{preview_str}]")

            # validation Loop
            target_tokens = []  # type: ignore
            while True:
                user_input = input(
                    f"       >> Enter prediction strings matching '{gt_name}' (comma-separated, or 'NONE'): "
                ).strip()

                if not user_input:
                    continue

                if user_input.upper() == "NONE":
                    target_tokens = []
                    break

                # Parse and Clean Input
                raw_tokens = [t.strip() for t in user_input.split(",") if t.strip()]

                # Check validity
                all_valid = True
                invalid_tokens = []
                pool_norm = {normalize_text(p) for p in available_preds}

                for t in raw_tokens:
                    if normalize_text(t) not in pool_norm:
                        all_valid = False
                        invalid_tokens.append(t)

                if all_valid:
                    target_tokens = raw_tokens
                    break
                else:
                    print(
                        f"          [!] Error: The following strings were NOT found in the pool: {invalid_tokens}"
                    )
                    print("              Please check for typos and try again.")

            # auto-Counting and removal
            found_count = 0
            if target_tokens:
                new_pool = []
                # Create a set of normalized target tokens for fast lookup
                target_norms = {normalize_text(t) for t in target_tokens}

                for item in available_preds:
                    if normalize_text(item) in target_norms:
                        found_count += 1
                        # We do not add it to new_pool (effectively removing it)
                    else:
                        new_pool.append(item)

                available_preds = new_pool
                print(
                    f"          -> Found {found_count} matches for {target_tokens}. Removed from pool."
                )
            else:
                print("          -> No matches selected.")

            # Update Metrics
            valid_matches = min(count, found_count)
            missed_matches = max(0, count - found_count)
            tp += valid_matches
            fn += missed_matches

        # low Volume Item (< 10)
        else:
            for i in range(count):
                if not available_preds:
                    fn += 1
                    continue

                print(f"\n       Item '{gt_name}' ({i + 1}/{count}):")
                for idx, p_item in enumerate(available_preds):
                    print(f"       [{idx + 1}] {p_item}")
                print("       [0] NO MATCH")

                sel = _ui_get_int(len(available_preds), "Match")

                if sel == 0:
                    fn += 1
                else:
                    tp += 1
                    available_preds.pop(sel - 1)

    # Any remaining predictions are False Positives
    fp = len(available_preds)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        0.0
        if (precision + recall) == 0
        else 2 * (precision * recall) / (precision + recall)
    )

    return precision, recall, f1


COLS_AUTO_NUM = ["grube_länge", "grube_breite", "grube_tiefe"]
COLS_AUTO_CAT = [
    "bestattung_geschlecht",
    "bestattung_alter",
    "bestattung_störung",
    "grube_form",
]
COLS_MANUAL = [
    "fundort",
    "bezirk",
    "bestattung_lage",
    "bestattung_ritus",
    "bestattung_orientierung",
    # "beigaben_liste", # Dont add beigaben_liste here, will be looked at automatically
    "grabeinbauten",
]


def compare_graves(
    gt_filepath: str, pred_filepath: str, output_csv_path: str, output_json_path: str
) -> Optional[pd.DataFrame]:
    try:
        df_gt = pd.read_csv(gt_filepath)
        df_pred = pd.read_csv(pred_filepath)
    except Exception as e:
        logger.error(f"Error loading files: {e}")
        return None

    cols_auto_numeric = [c for c in COLS_AUTO_NUM if c in df_gt.columns]
    cols_auto_cat = [c for c in COLS_AUTO_CAT if c in df_gt.columns]
    cols_manual = [c for c in COLS_MANUAL if c in df_gt.columns]
    cols_auto = cols_auto_numeric + cols_auto_cat

    df_result = df_gt.copy()

    df_pred_work = df_pred.copy()
    df_pred_work["page_id"] = df_pred_work["page_id"].astype(str)
    df_pred_work["grab_identifikation"] = df_pred_work["grab_identifikation"].astype(
        str
    )

    summary_no_match = []
    summary_duplicates = []
    summary_multi_bestattung = []  # type: ignore
    summary_hallucinations: dict[str, list[int]] = {}
    summary_omissions: dict[str, list[int]] = {}

    summary_matches = []

    tuple_stats = {}  # type: ignore

    # iterate
    for idx_gt, row_gt in df_gt.iterrows():
        page_id = str(row_gt["page_id"])
        grab_id = str(row_gt["grab_identifikation"])
        row_id_str = f"Idx {idx_gt} ('{grab_id}')"

        _ui_print_grave_header(grab_id, page_id, idx_gt)  # type: ignore

        # Matching
        # subset preserves the original indices (global indices) of df_pred_work
        subset = df_pred_work[df_pred_work["page_id"] == page_id]

        if subset.empty:
            logger.error(f"GT Index {idx_gt}: Page {page_id} not found.")
            summary_no_match.append(idx_gt)
            _set_row_null(df_result, idx_gt, cols_auto, cols_manual)
            if "beigaben_liste" in df_result.columns:
                df_result.at[idx_gt, "beigaben_liste"] = None  # type: ignore
            continue

        matches = subset[
            subset["grab_identifikation"].apply(normalize_text)
            == normalize_text(grab_id)
        ]

        row_pred = None

        if len(matches) == 1:
            row_pred = matches.iloc[0]

        else:
            # Ambiguous match - Ask User
            candidates = subset.copy()
            c_ids = candidates["grab_identifikation"].tolist()
            c_section_titles = (
                candidates["section_title"].tolist()
                if "section_title" in candidates.columns
                else ["" for _ in range(len(candidates))]
            )
            candidate_indices = candidates.index.tolist()

            opt_multiple = len(c_ids) + 1

            print("   [?] ROW MATCHING: Ambiguous candidates found")
            for i, (c_id, c_title, c_idx) in enumerate(
                zip(c_ids, c_section_titles, candidate_indices)
            ):
                print(
                    f"       [{i + 1}] {c_id} ('{c_title}') (Extract DF Idx: {c_idx})"
                )
            print("       [0] NO MATCH")
            print(f"       [{opt_multiple}] MULTIPLE MATCHES (Treat as Duplicate)")

            sel = _ui_get_int(opt_multiple, "Link to")

            if sel == 0:
                summary_no_match.append(idx_gt)
                _set_row_null(df_result, idx_gt, cols_auto, cols_manual)
                if "beigaben_liste" in df_result.columns:
                    df_result.at[idx_gt, "beigaben_liste"] = None  # type: ignore
                continue
            elif sel == opt_multiple:
                summary_duplicates.append(idx_gt)
                _set_row_null(df_result, idx_gt, cols_auto, cols_manual)
                if "beigaben_liste" in df_result.columns:
                    df_result.at[idx_gt, "beigaben_liste"] = None  # type: ignore
                continue
            else:
                row_pred = candidates.iloc[sel - 1]

        # Compare
        if row_pred is not None:
            summary_matches.append([int(idx_gt), int(row_pred.name)])  # type: ignore

            _fill_comparison(
                df_result,
                idx_gt,
                row_gt,
                row_pred,
                df_pred_work,
                cols_auto_numeric,
                cols_auto_cat,
                cols_manual,
                summary_multi_bestattung,
                summary_hallucinations,
                summary_omissions,
                tuple_stats,
                row_id_str,
            )

    # output and metrics
    try:
        df_result.to_csv(output_csv_path, index=False)
        logger.info(f"Saved result: {output_csv_path}")
    except Exception as e:
        logger.error(f"Save failed: {e}")

    metrics_per_col = {}
    total_score_sum = 0.0
    total_valid_count = 0

    compared_cols = set(cols_auto_numeric + cols_auto_cat + cols_manual)
    if "beigaben_liste" in df_result.columns:
        compared_cols.add("beigaben_liste")
    eval_cols = [c for c in df_result.columns if c in compared_cols]

    for col in eval_cols:
        col_data = pd.to_numeric(df_result[col], errors="coerce")
        valid_mask = col_data.notna()
        valid_count = int(valid_mask.sum())

        if valid_count == 0:
            metrics_per_col[col] = {"metric_score": None, "score_sum": 0, "total": 0}
            continue

        score_sum = float(col_data[valid_mask].sum())
        avg_score = score_sum / valid_count

        metrics_per_col[col] = {
            "metric_score": round(avg_score, 4),  # type: ignore
            "score_sum": round(score_sum, 4),  # type: ignore
            "total": valid_count,
        }
        total_score_sum += score_sum
        total_valid_count += valid_count

    global_acc = (total_score_sum / total_valid_count) if total_valid_count > 0 else 0

    tuple_json = {}
    for col, data in tuple_stats.items():
        lengths = data["lengths"]
        indices = data["indices"]
        count = len(lengths)
        avg_len = sum(lengths) / count if count > 0 else 0
        tuple_json[col] = {"count": count, "avg_len": avg_len, "gt_indices": indices}
    json_data = {
        "metadata": {
            "ground_truth_file": gt_filepath,
            "prediction_file": pred_filepath,
            "comparison_matrix_file": output_csv_path,
            "timestamp": datetime.now().isoformat(),
        },
        "matches": summary_matches,
        "statistics": {
            "no_matches": {"count": len(summary_no_match), "items": summary_no_match},
            "duplicate_matches": {
                "count": len(summary_duplicates),
                "items": summary_duplicates,
            },
            "tuple_errors": tuple_json,
            "hallucinations_per_column": summary_hallucinations,
            "omissions_per_column": summary_omissions,
        },
        "metrics": {
            "global_performance": round(global_acc, 4),
            "total_score_sum": round(total_score_sum, 4),
            "total_valid_comparisons": total_valid_count,
            "per_column": metrics_per_col,
        },
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    logger.info(f"Global Acc: {global_acc:.2%}")
    return df_result


def _set_row_null(
    df: pd.DataFrame, idx: Any, cols_auto: list[str], cols_manual: list[str]
) -> None:
    """Sets all comparison columns to None (Skipped)."""
    df.loc[idx, cols_auto] = None
    df.loc[idx, cols_manual] = None


def _get_manual_score(col: str, gt: Any, pred: Any) -> int:
    # Handle both being NaN (Match)
    if pd.isna(gt) and pd.isna(pred):
        return 1

    # Normalize strings for comparison
    # Treat NaN as empty string for the string comparison check
    gt_str = str(gt) if pd.notna(gt) else ""
    pred_str = str(pred) if pd.notna(pred) else ""

    # Silent Check: Exact string matches (normalized)
    if normalize_text(gt_str) == normalize_text(pred_str):
        return 1

    # Active Prompt
    # We reach here if they don't match textually (including NaN vs "Value")
    print(f"\n   [?] CHECK: {col}")
    print(f"       GT   : {gt}")
    print(f"       Pred : {pred}")

    return _ui_get_bool("Correct?")


def _fill_comparison(
    df_result: pd.DataFrame,
    idx_gt: Any,
    row_gt: pd.Series,
    row_pred: pd.Series,
    df_pred: pd.DataFrame,
    cols_num: list[str],
    cols_cat: list[str],
    cols_manual: list[str],
    summary_multi: list[str],
    summary_hallu: dict[str, list[int]],
    summary_omissions: dict[str, list[int]],
    tuple_stats: dict[str, dict[str, list]],
    row_id_str: str,
) -> None:
    def _parse_val(val: Any) -> tuple[Any, bool]:
        """
        Returns (parsed_value, is_tuple_bool).
        parsed_value is the tuple if is_tuple_bool is True, else the original value.
        """
        if isinstance(val, str) and val.strip().startswith("(") and ")" in val:
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, (tuple, list)):
                    return parsed, True
            except (ValueError, SyntaxError):
                pass
        if isinstance(val, (tuple, list)):
            return val, True
        return val, False

    def _handle_tuple_penalty(col: str, val: Any) -> bool:
        length = len(val) if isinstance(val, (tuple, list)) else 0
        if col not in tuple_stats:
            tuple_stats[col] = {"lengths": [], "indices": []}

        tuple_stats[col]["lengths"].append(length)
        tuple_stats[col]["indices"].append(int(idx_gt))
        summary_multi.append(row_id_str)
        # Prediction is ambiguous (tuple), strictly mark as 0
        df_result.at[idx_gt, col] = 0
        return True

    # manual columns
    for col in cols_manual:
        val_gt = row_gt[col]
        raw_pred = row_pred.get(col, None)

        # Parse Prediction
        val_pred, pred_is_tuple = _parse_val(raw_pred)

        # Parse GT (check if it contains a list of valid options)
        val_gt_parsed, gt_is_tuple = _parse_val(val_gt)

        # handle Tuples in Prediction (Ambiguity Penalty)
        if pred_is_tuple:
            if _handle_tuple_penalty(col, val_pred):
                # If GT is None and Pred is Tuple -> Hallucination
                if pd.isna(val_gt):
                    summary_hallu.setdefault(col, []).append(int(idx_gt))
                continue

        # omission: GT is Present, Pred is Missing
        if pd.notna(val_gt) and pd.isna(val_pred):
            df_result.at[idx_gt, col] = 0
            summary_omissions.setdefault(col, []).append(int(idx_gt))
            continue

        is_match = False

        # Case: Both are None
        if pd.isna(val_gt) and pd.isna(val_pred):
            is_match = True

        # Case: Comparison needed
        elif pd.notna(val_gt) and pd.notna(val_pred):
            norm_pred = normalize_text(str(val_pred))

            if gt_is_tuple:
                # GT is a tuple: Check if normalized Pred exists in GT list
                for item in val_gt_parsed:
                    if normalize_text(str(item)) == norm_pred:
                        is_match = True
                        break
            else:
                # GT is a single string: Direct comparison
                if normalize_text(str(val_gt_parsed)) == norm_pred:
                    is_match = True

        if is_match:
            res = 1
        else:
            res = _get_manual_score(col, val_gt, val_pred)

        df_result.at[idx_gt, col] = res

        # hallucination check
        if res == 0:
            if pd.isna(val_gt) and pd.notna(raw_pred):
                summary_hallu.setdefault(col, []).append(int(idx_gt))

    # auto numeric
    for col in cols_num:
        val_gt_raw = row_gt[col]
        raw_pred = row_pred.get(col, None)

        # Parse Prediction
        val_pred, pred_is_tuple = _parse_val(raw_pred)

        # Parse Ground Truth (Check for tuple of valid answers)
        val_gt_parsed, gt_is_tuple = _parse_val(val_gt_raw)

        if pd.isna(val_gt_raw) and pd.notna(raw_pred):
            summary_hallu.setdefault(col, []).append(int(idx_gt))

        if pd.notna(val_gt_raw) and pd.isna(raw_pred):
            summary_omissions.setdefault(col, []).append(int(idx_gt))

        if pred_is_tuple:
            if _handle_tuple_penalty(col, val_pred):
                continue

        # Logic: If GT is tuple, check if pred matches ANY of them
        match_found = False

        # Case 1: Both NaN
        if pd.isna(val_gt_raw) and pd.isna(val_pred):
            match_found = True
        # Case 2: One NaN, one not
        elif pd.isna(val_gt_raw) or pd.isna(val_pred):
            match_found = False
        else:
            try:
                pred_float = float(val_pred)

                if gt_is_tuple:
                    # GT is tuple: Check if ANY item in GT is close to pred
                    for gt_item in val_gt_parsed:
                        try:
                            if abs(float(gt_item) - pred_float) < 0.1:
                                match_found = True
                                break
                        except (ValueError, TypeError):
                            continue
                else:
                    # GT is single value
                    val_gt = float(val_gt_raw)
                    if abs(val_gt - pred_float) < 0.1:
                        match_found = True
            except Exception:
                # Fallback to manual if float conversion fails
                res = _get_manual_score(col, val_gt_raw, val_pred)
                df_result.at[idx_gt, col] = res
                continue

        df_result.at[idx_gt, col] = 1 if match_found else 0

    # auto categorical
    for col in cols_cat:
        val_gt_raw = row_gt[col]
        raw_pred = row_pred.get(col, None)

        # Parse Prediction
        val_pred, pred_is_tuple = _parse_val(raw_pred)

        # Parse Ground Truth
        val_gt_parsed, gt_is_tuple = _parse_val(val_gt_raw)

        if pd.isna(val_gt_raw) and pd.notna(raw_pred):
            summary_hallu.setdefault(col, []).append(int(idx_gt))

        if pd.notna(val_gt_raw) and pd.isna(raw_pred):
            summary_omissions.setdefault(col, []).append(int(idx_gt))

        if pred_is_tuple:
            if _handle_tuple_penalty(col, val_pred):
                continue

        match_found = False

        if pd.isna(val_gt_raw) and pd.isna(val_pred):
            match_found = True
        elif pd.isna(val_gt_raw) or pd.isna(val_pred):
            match_found = False
        else:
            try:
                if gt_is_tuple:
                    # GT is tuple: Check if ANY item matches pred
                    for gt_item in val_gt_parsed:
                        # Numeric comparison inside categorical list (rare but possible)
                        if isinstance(val_pred, (float, int)) and isinstance(
                            gt_item, (float, int)
                        ):
                            if gt_item == val_pred:
                                match_found = True
                                break
                        # String comparison
                        elif str(gt_item) and str(val_pred):
                            if normalize_text(str(gt_item)) == normalize_text(
                                str(val_pred)
                            ):
                                match_found = True
                                break
                else:
                    # GT is single value
                    if isinstance(val_pred, (float, int)) and isinstance(
                        val_gt_parsed, (float, int)
                    ):
                        match_found = val_gt_parsed == val_pred
                    else:
                        match_found = normalize_text(
                            str(val_gt_parsed)
                        ) == normalize_text(str(val_pred))
            except Exception:
                res = _get_manual_score(col, val_gt_raw, val_pred)
                df_result.at[idx_gt, col] = res
                continue

        df_result.at[idx_gt, col] = 1 if match_found else 0

    if "beigaben_liste" in df_result.columns:
        col = "beigaben_liste"
        val_gt = row_gt.get(col, None)
        val_pred = row_pred.get(col, None)
        val_gt_str = str(val_gt) if pd.notna(val_gt) else ""
        val_pred_str = str(val_pred) if pd.notna(val_pred) else ""

        if not val_gt_str.strip() and val_pred_str.strip():
            summary_hallu.setdefault(col, []).append(int(idx_gt))

        if val_gt_str.strip() and not val_pred_str.strip():
            summary_omissions.setdefault(col, []).append(int(idx_gt))

        _, _, f1 = _calculate_beigaben_f1(val_gt_str, val_pred_str)
        df_result.at[idx_gt, col] = f1
