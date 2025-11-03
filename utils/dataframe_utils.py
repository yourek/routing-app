import random
import string
from datetime import datetime, time
from io import BytesIO

from typing import Tuple

import pandas as pd
from db.db_operations import upsert_delete
from params.time_allocation import AVAILABLE_VISIT_FREQUENCIES
import streamlit as st
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import hashlib
import warnings


warnings.filterwarnings("ignore", message="Could not infer format")


def df_hash(df: pd.DataFrame) -> str:
    """Compact hash representation of dataframe contents"""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


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
        return df  # , None, False

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

    with st.container(border=True):
        st.markdown("#### 🔍 Filter data")

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


def batch_apply_schedule_adjustments(
    df: pd.DataFrame,
    scen_metadata: dict | None = None,
) -> pd.DataFrame:

    if_adj_submitted = False
    schedule_adjustments = None

    if scen_metadata:
        with st.expander(
            "Batch time allocations updater (updates for all filtered rows)"
        ):
            role_names = [role["name"] for role in scen_metadata["roles"]]
            # st.markdown(
            #     """
            #     <style>
            #     /* Make number input boxes narrower */
            #     div.stNumberInput > div {
            #         max-width: 180px;  /* adjust as needed */
            #     }
            #     div.stSelectbox > div {
            #         max-width: 180px;  /* adjust width as needed */
            #     }
            #     </style>
            #     """,
            #     unsafe_allow_html=True,
            # )
            cols = st.columns([3, 3, 3, 2])
            with cols[0]:
                selected_role = st.selectbox("Role", role_names)
            with cols[1]:
                visits_nb = st.selectbox(
                    "Visits per month [#]", options=AVAILABLE_VISIT_FREQUENCIES
                )
            with cols[2]:
                duration = st.number_input(
                    "Time in store [hours]",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=1,
                )
            with cols[3]:
                st.markdown("<br>", unsafe_allow_html=True)
                if_adj_submitted = st.button(
                    "✅ Apply adjustments", key="adjustment-button"
                )
                # if if_adj_submitted:
                #     st.warning(
                #         f"Are you sure you to modify visits number and time in store?"
                #     )
                #     col1, col2 = st.columns(2)
                #     a, b = False, False
                #     with col1:
                #         a = st.button("✅ Yes, modify")
                #     with col2:
                #         b = st.button("❌ Cancel")
                #     print(a, b)

        if if_adj_submitted:
            schedule_adjustments = {
                "selected_role": selected_role,
                "visits_nb": visits_nb,
                "duration": duration,
                "indexes_to_modify": df.index,
            }
        else:
            schedule_adjustments = {"indexes_to_modify": df.index}

    return if_adj_submitted, schedule_adjustments


def render_editable_table(
    df,
    column_config,
    update_key,
    data_path,
    key_pattern=None,
    cols_to_hide=None,
    column_order=None,
):
    ## HEIGHT
    ROW_HEIGHT = 35
    HEADER_HEIGHT = 40
    MAX_TABLE_HEIGHT = 550
    dynamic_height = min(HEADER_HEIGHT + len(df) * ROW_HEIGHT, MAX_TABLE_HEIGHT)
    ##

    entry_df_hash = df_hash(df)
    session_state_key = f"changes_made_{entry_df_hash}"
    editor_key_name = f"editor_key_{key_pattern}"

    editor_key = st.session_state.get(editor_key_name, entry_df_hash)

    if not key_pattern:
        key_pattern = gen_randkey(6)

    if not cols_to_hide:
        cols_to_hide = []

    if not column_order:
        column_order = df.columns.tolist()

    cols_to_show = [c for c in column_order if c not in cols_to_hide]

    editable_table = st.data_editor(
        df,
        key=editor_key,
        column_config=column_config,
        column_order=cols_to_show,
        height=dynamic_height,
    )
    output_df_hash = df_hash(editable_table)

    # st.write(f"**Data entry hash:** {entry_df_hash}")

    if output_df_hash != entry_df_hash:
        st.session_state[session_state_key] = True
    else:
        st.session_state[session_state_key] = False

    if st.session_state.get(session_state_key, False):
        c_save, c_reset, spacer = st.columns([1, 1, 5], gap="small")
        if c_save.button("Save changes", type="primary", key=f"save_{key_pattern}"):
            upsert_delete(
                edited_df=editable_table,
                json_path=data_path,
                key_col=update_key,
                # deleted_keys=deleted_keys,
            )
            st.session_state[session_state_key] = False

            if session_state_key in st.session_state:
                del st.session_state[session_state_key]
            if "editor_key" in st.session_state:
                del st.session_state["editor_key"]

            st.success("Changes saved.")
            st.rerun()
        if c_reset.button("Cancel", key=f"cancel_{key_pattern}"):
            st.session_state[session_state_key] = False
            st.session_state[editor_key_name] = (
                f"{entry_df_hash}_{st.session_state.get('reset_counter', 0)+1}"  # new key
            )
            st.session_state["reset_counter"] = (
                st.session_state.get("reset_counter", 0) + 1
            )
            st.rerun()

    # filtered = {
    #     k: v
    #     for k, v in st.session_state.items()
    #     if ("changes_made" in k) or ("editor_key" in k)
    # }
    # st.write(filtered)
    # st.write(f"**Data output hash:** {output_df_hash}")


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
    # for col, dtype in column_dtypes.items():
    #     if dtype == "bool":  # Skip bool as it defaults to False
    #         continue
    #     if col in changed and (changed[col].isna().any() or changed[col].eq("").any()):
    #         st.error(f"Some rows in '{col}' column have empty values.")
    #         raise TypeError("Empty values")

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


def download_dataframe(
    df: pd.DataFrame, output_file_name: str, button_label="Download as XLSX"
):
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx_data = to_excel_bytes(df)
        st.download_button(
            label=button_label,
            data=xlsx_data,
            file_name=f"{output_file_name}_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        st.error(f"Error preparing download: {e}")


def validate_visits_number(df):
    visits_nb_columns = [c for c in df.columns if " - visits per month" in c]
    visits_nb_correct = (
        df[visits_nb_columns].isin(AVAILABLE_VISIT_FREQUENCIES).all().all()
    )

    return visits_nb_correct


def sidebar_load_download_section_from_session_state(
    dfs_session_key, file_path, project_id, update_columns_key, column_dtypes
):
    st.subheader("Load / Save Time Allocation Data")

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = "uploader_time_alloc"

    load_mode = st.radio(
        "Load mode",
        options=["Replace", "Update"],
        index=0,
        horizontal=True,
        key="load_mode_choice",
        help="Replace -> overwrites all rows \n\n Update -> upserts rows by Customer ID",
    )

    uploaded_file = st.file_uploader(
        "Upload XLSX file",
        type=["xlsx"],
        key=st.session_state["uploader_key"],
    )

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            df_up = pd.read_excel(
                uploaded_file,
                sheet_name="Data",
                engine="openpyxl",
                dtype=column_dtypes,
            )
            df_up = validate_and_align_columns(df_up, column_dtypes)
            visits_nb_correct = validate_visits_number(df_up)
            if not visits_nb_correct:
                st.error(
                    f"Visits number columns need to have values from range: {str(AVAILABLE_VISIT_FREQUENCIES)}!"
                )
                return
            df_up = df_up.sort_values(by=update_columns_key).reset_index(drop=True)
            if load_mode == "Replace":
                st.session_state[dfs_session_key] = df_up
                df_up.to_json(file_path, orient="records", indent=2)
                st.success("Loaded XLSX file successfully.")
            else:
                result_df = upsert_delete(
                    edited_df=df_up,
                    json_path=file_path,
                    key_col=update_columns_key,
                    deleted_keys=None,
                )
                st.session_state[dfs_session_key] = result_df
                st.success("Updated table from XLSX successfully.")
        except Exception as e:
            st.error(f"Error reading XLSX: {e}")

    download_dataframe(
        st.session_state[dfs_session_key],
        project_id,
        "Download Time Allocation data as XLSX",
    )

    # # XLSX download -> ALWAYS from canonical DF in session
    # xlsx_data = to_excel_bytes(st.session_state["stores_w_role_spec"])
    # st.download_button(
    #     label="Download current table as XLSX",
    #     data=xlsx_data,
    #     file_name=f"{project_id}_{ts}.xlsx",
    #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    # )


def convert_date_types_to_string(df):
    df = df.copy()
    df = df.astype(
        {
            col: "string"
            for col in df.select_dtypes(include=["datetime", "datetimetz"]).columns
        }
    )

    return df
