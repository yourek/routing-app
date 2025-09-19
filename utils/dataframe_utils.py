import random
import string
from datetime import time
from io import BytesIO

import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


def filter_dataframe(df: pd.DataFrame, enable_filters: bool) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns,
    including:
      - Boolean tri-state,
      - Numeric comparators with int-aware widgets,
      - Datetime range,
      - Object/text columns with dropdown of unique values.
    """
    if not enable_filters:
        return df

    df = df.copy()

    # Normalize datetime-like columns (optional)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    with st.container():
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))

            # Boolean
            if is_bool_dtype(df[column]):
                choice = right.selectbox(
                    f"Value for {column}",
                    options=["Any", "True", "False"],
                    index=0,
                    key=f"bool_{column}",
                )
                if choice == "True":
                    df = df[df[column] == True]
                elif choice == "False":
                    df = df[df[column] == False]

            # Categorical
            elif is_categorical_dtype(df[column]):
                opts = pd.Series(df[column].unique()).dropna().tolist()
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    options=opts,
                    default=opts,
                    key=f"cat_{column}",
                )
                if user_cat_input:
                    df = df[df[column].isin(user_cat_input)]
                else:
                    df = df.iloc[0:0]

            # Numeric (int-aware)
            elif is_numeric_dtype(df[column]):
                comp = right.selectbox(
                    f"Filter {column} by",
                    options=["between", "=", ">", ">=", "<", "<="],
                    index=0,
                    key=f"num_comp_{column}",
                )

                # Work with a numeric Series; keep track if it’s integer dtype
                series = pd.to_numeric(df[column], errors="coerce")
                is_int = is_integer_dtype(df[column])

                s_min, s_max = series.min(), series.max()
                is_int = is_integer_dtype(df[column])

                if is_int:
                    col_min = int(s_min)
                    col_max = int(s_max) if s_min != s_max else int(s_max) + 1
                else:
                    col_min = float(s_min)
                    col_max = float(s_max) if s_min != s_max else float(s_max) + 1e-9

                if comp == "between":
                    if is_int:
                        low, high = right.slider(
                            f"Values for {column}",
                            min_value=col_min,
                            max_value=col_max,
                            value=(col_min, col_max),
                            step=1,
                            key=f"num_between_{column}",
                        )
                    else:
                        step = (col_max - col_min) / 100 if col_max > col_min else 1.0
                        low, high = right.slider(
                            f"Values for {column}",
                            min_value=col_min,
                            max_value=col_max,
                            value=(col_min, col_max),
                            step=step,
                            key=f"num_between_{column}",
                        )
                    df = df[series.between(low, high)]
                else:
                    if is_int:
                        val = right.number_input(
                            f"Value for {column}",
                            value=col_min,
                            step=1,
                            format="%d",
                            key=f"num_value_{column}",
                        )
                        val = int(val)
                    else:
                        val = right.number_input(
                            f"Value for {column}",
                            value=col_min,
                            key=f"num_value_{column}",
                        )

                    if comp == "=":
                        df = df[series == val]
                    elif comp == ">":
                        df = df[series > val]
                    elif comp == ">=":
                        df = df[series >= val]
                    elif comp == "<":
                        df = df[series < val]
                    elif comp == "<=":
                        df = df[series <= val]

            # Datetime
            elif is_datetime64_any_dtype(df[column]):
                start_default = pd.to_datetime(df[column].min()).date()
                end_default = pd.to_datetime(df[column].max()).date()
                user_date_input = right.date_input(
                    f"Range for {column}",
                    value=(start_default, end_default),
                    key=f"date_{column}",
                )
                if (
                    isinstance(user_date_input, (list, tuple))
                    and len(user_date_input) == 2
                ):
                    start_date, end_date = map(pd.to_datetime, user_date_input)
                    df = df.loc[df[column].between(start_date, end_date)]

            # Object/text: dropdown
            else:
                unique_vals = pd.Series(
                    df[column].dropna().astype(str).unique()
                ).tolist()
                selected_vals = right.multiselect(
                    f"Select values in {column}",
                    options=unique_vals,
                    default=[],
                    key=f"obj_multi_{column}",
                    help="Type to search and select one or more exact values.",
                )
                if selected_vals:
                    df = df[df[column].astype(str).isin(selected_vals)]

    return df


def gen_randkey(n=8):
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def load_from_json(stores_json_path, column_dtypes) -> pd.DataFrame:
    if stores_json_path.exists():
        return pd.read_json(stores_json_path, orient="records")
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in column_dtypes.items()})


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return output.getvalue()


def get_deleted_rows_keys(df: pd.DataFrame, key_col: str, editor_key: str) -> list[str]:
    state = st.session_state.get(editor_key, {})
    deleted_pos = state.get("deleted_rows", [])
    mapped_df = df.reset_index(drop=True)

    return mapped_df.loc[deleted_pos, key_col].astype(str).tolist()


def validate_new_row_addition(changed, id_col, column_dtypes):
    # 1) Pull newly added rows from the editor's session state (list of dicts)
    editor_state = st.session_state.get(st.session_state["stores_editor_key"], {})
    added_rows = editor_state.get("added_rows", []) or []

    # 2) Extract IDs from added rows only (string-normalized)
    added_ids = []
    for r in added_rows:
        v = r.get(id_col, None)
        if pd.notna(v) and str(v).strip() != "":
            added_ids.append(str(v).strip())

    # 3) Build the original ID set from the full, unfiltered canonical table
    orig_ids = set(
        st.session_state["stores_table"][id_col].astype(str)
    )  # full canonical, not filtered

    # 4) Offenders: any added ID already present in the original canonical data
    offenders = sorted([cid for cid in set(added_ids) if cid in orig_ids])

    if offenders:
        st.error(
            f"Save blocked: attempted to add new row(s) with an existing Customer ID. "
            "Modify the existing row instead. {id_col} values: " + ", ".join(offenders)
        )
        raise ValueError(f"Duplicate {id_col}(s) on insert: {', '.join(offenders)}")

    # Empty column validation
    for col, dtype in column_dtypes.items():
        if dtype == "bool":  # Skip bool as it defaults to False
            continue
        if col in changed and (changed[col].isna().any() or changed[col].eq("").any()):
            st.error(f"Some rows in '{col}' column have empty values.")
            raise TypeError("Empty values")

    for col, dtype in column_dtypes.items():
        changed[col] = changed[col].astype(dtype)


def validate_and_align_columns(df: pd.DataFrame, column_dtypes: dict) -> pd.DataFrame:
    # Keep only allowed columns, in defined order
    cols = list(column_dtypes.keys())
    df = df.reindex(columns=[c for c in df.columns if c in column_dtypes])
    # Add any missing columns
    for c, dt in column_dtypes.items():
        if c not in df.columns:
            if dt == "bool":
                df[c] = pd.Series([pd.NA] * len(df), dtype="boolean")
            elif dt == "float":
                df[c] = pd.Series([pd.NA] * len(df), dtype="float64")
            else:
                df[c] = pd.Series([pd.NA] * len(df), dtype="string")
    # Reorder columns to dict order
    df = df.loc[:, cols]

    return df


def to_time(hhmm: str, fallback: time) -> time:
    try:
        h, m = map(int, hhmm.split(":"))
        return time(h, m)
    except Exception:
        return fallback
