from pathlib import Path
import pandas as pd
from typing import Iterable


def upsert_delete(
    edited_df: pd.DataFrame,
    json_path: Path,
    key_col: str = "Customer",
    deleted_keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    if json_path.exists() and json_path.stat().st_size > 0:
        try:
            existing_df = pd.read_json(json_path, orient="records")
        except ValueError:
            existing_df = pd.DataFrame(columns=edited_df.columns)
    else:
        existing_df = pd.DataFrame(columns=edited_df.columns)

    if key_col not in edited_df.columns:
        raise KeyError(f"Key column '{key_col}' not found in edited_df")

    if key_col not in existing_df.columns:
        existing_df = existing_df.reindex(columns=edited_df.columns)

    # Normalize keys
    edited_df = edited_df.copy()
    edited_df[key_col] = edited_df[key_col].astype(str)
    existing_df[key_col] = existing_df[key_col].astype(str)

    # 1) Apply explicit deletions only
    if deleted_keys:
        deleted_keys = {str(k) for k in deleted_keys}
        existing_df = existing_df[~existing_df[key_col].isin(deleted_keys)]

    # 2) Upsert edited rows
    replace_keys = set(edited_df[key_col].astype(str))
    base = existing_df[~existing_df[key_col].isin(replace_keys)]
    result = pd.concat([base, edited_df], ignore_index=True)
    result = result.drop_duplicates(subset=[key_col], keep="last").reset_index(
        drop=True
    )
    result = result.sort_values(by=key_col).reset_index(drop=True)

    result.to_json(json_path, orient="records", indent=2)
    return result
