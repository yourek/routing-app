from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

from session.session import init_session
from utils.auth_utils.guards import require_active_project, require_authentication
from utils.dataframe_utils import (
    filter_dataframe,
    gen_randkey,
    get_deleted_rows_keys,
    to_excel_bytes,
    load_from_json,
    validate_new_row_addition,
    validate_and_align_columns,
)
from db.db_operations import upsert_delete

init_session()
require_authentication()
require_active_project()

active_project = st.session_state.get("active_project")
project_name = active_project["name"]
project_id = active_project["id"]
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

project_data_path = Path("data") / "projects" / project_id
project_data_path.mkdir(parents=True, exist_ok=True)
stores_json_path = project_data_path / "stores.json"

COLUMN_DTYPES = {
    "Customer": "str",
    "Stores": "str",
    "Chain": "str",
    "City": "str",
    "Street": "str",
    "Channel": "str",
    "Group": "str",
    "Grading": "str",
    "Sales Avg": "float",
    "Latitude": "float",
    "Longitude": "float",
    "Sunday": "bool",
    "Monday": "bool",
    "Tuesday": "bool",
    "Wednesday": "bool",
    "Thursday": "bool",
    "Friday": "bool",
    "Saturday": "bool",
}

ID_COL = "Customer"
TABLE_HEIGHT = 550

# -----------------------------
# KEY INIT AND CLEAR
# -----------------------------
# Clear stores after detecting project change
if "last_project_id" not in st.session_state:
    st.session_state["last_project_id"] = project_id

if st.session_state["last_project_id"] != project_id:
    for k in ["stores_table", "stores_table_work", "stores_editor_key"]:
        if k in st.session_state:
            del st.session_state[k]
    # Update tracker
    st.session_state["last_project_id"] = project_id

# Initialize canonical DF
if "stores_table" not in st.session_state:
    base = load_from_json(stores_json_path, COLUMN_DTYPES)
    st.session_state["stores_table"] = validate_and_align_columns(base, COLUMN_DTYPES)

if "stores_editor_key" not in st.session_state:
    st.session_state["stores_editor_key"] = gen_randkey()

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = "uploader_0"

if "show_editor" not in st.session_state:
    st.session_state["show_editor"] = False

# Handle switch to Table View
if st.session_state.get("switch_to_view"):
    st.session_state["show_editor"] = False  # set widget value pre-instantiation
    st.session_state.pop("switch_to_view", None)  # clear the control flag
    st.rerun()  # ensure the next run renders the else branch


st.title("🏬 Add & Manage Stores")
st.caption(f"for {project_name}")

column_config = {
    col: st.column_config.Column(col, width="auto") for col in COLUMN_DTYPES.keys()
}

# -----------------------------
# SIDEBAR LOAD / SAVE
# -----------------------------
with st.sidebar:
    st.subheader("Load / Save")

    st.session_state["uploader_key"] = "uploader_stores"

    # Choose load mode: Replace (overwrite) or Update (upsert)
    load_mode = st.radio(
        "Load mode",
        options=["Replace", "Update"],
        index=0,
        horizontal=True,
        key="load_mode_choice",
        help="Replace -> overwrites all rows \n\n Update -> upserts rows by Customer ID",
    )

    uploaded_file = st.file_uploader(
        "Upload XLSX file", type=["xlsx"], key=st.session_state["uploader_key"]
    )

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            df_up = pd.read_excel(
                uploaded_file,
                sheet_name="Data",
                engine="openpyxl",
                keep_default_na=True,
                na_values=["NA", ""],
                dtype=COLUMN_DTYPES,
            )
            weekdays = [
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
            ]
            for weekday in weekdays:
                if weekday in df_up.columns:
                    df_up[weekday] = df_up[weekday].fillna(False)
                else:
                    df_up[weekday] = False
            df_up = validate_and_align_columns(df_up, COLUMN_DTYPES)
            df_up = df_up.sort_values(by=ID_COL).reset_index(drop=True)

            if load_mode == "Replace":
                st.session_state["stores_table"] = df_up  # update canonical
                df_up.to_json(stores_json_path, orient="records", indent=2)
                st.success("Loaded XLSX file successfully.")
            else:
                result_df = upsert_delete(
                    edited_df=df_up,
                    json_path=stores_json_path,
                    key_col=ID_COL,
                    deleted_keys=None,  # no deletions during bulk update
                )
                st.session_state["stores_table"] = result_df
                st.success("Updated table from XLSX successfully.")

            st.session_state["upload_processed"] = (
                True  # mark processed and rotate key to clear uploader
            )
            st.session_state["uploader_key"] = f"uploader_{datetime.now().timestamp()}"
        except Exception as e:
            st.error(f"Error reading XLSX: {e}")

    # Reset the 'processed' flag when page shows with no file selected
    if uploaded_file is None and st.session_state.get("upload_processed", False):
        st.session_state["upload_processed"] = False  # ready for next upload [16]

    # XLSX download -> ALWAYS from canonical DF in session
    xlsx_data = to_excel_bytes(st.session_state["stores_table"])
    st.download_button(
        label="Download current table as XLSX",
        data=xlsx_data,
        file_name=f"{project_id}_{ts}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# -----------------------------
# EDITOR BRANCH
# -----------------------------
if st.session_state["show_editor"]:
    st.title("Table Edit")
    # Feed editor from canonical DF directly to avoid divergence
    table = st.session_state["stores_table"].copy()

    # Buttons
    table_toggle = st.columns([1])[0]
    with table_toggle:
        show_editor = st.toggle("Show editable table", value=True, key="show_editor")

    filter_toggle, c_save, c_reset, spacer = st.columns([1, 1, 1, 7], gap="small")
    with filter_toggle:
        if ("apply_filters" not in st.session_state) or (
            st.session_state["apply_filters"] is False
        ):
            st.toggle("Apply filters", key="apply_filters", value=False)
        else:
            st.toggle("Apply filters", key="apply_filters", value=True)
    with c_save:
        save_clicked = st.button("Save changes", type="primary")
    with c_reset:
        reset_clicked = st.button("Reset changes")

    enable_filters = st.session_state.get("apply_filters", False)

    edited = st.data_editor(
        filter_dataframe(table, enable_filters),
        column_config=column_config,
        num_rows="dynamic",
        key=st.session_state["stores_editor_key"],
        height=TABLE_HEIGHT,
    )

    def clean_value(x):
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return x

    edited["Customer"] = edited["Customer"].apply(clean_value)

    if save_clicked:
        try:
            # breakpoint()
            changed = edited.copy()

            validate_new_row_addition(changed, ID_COL, COLUMN_DTYPES)

            deleted_keys = get_deleted_rows_keys(
                table, ID_COL, st.session_state["stores_editor_key"]
            )
            result_df = upsert_delete(
                edited_df=changed,
                json_path=stores_json_path,
                key_col=ID_COL,
                deleted_keys=deleted_keys,
            )
            result_df = validate_and_align_columns(result_df, COLUMN_DTYPES)

            # Update canonical DF
            st.session_state["stores_table"] = result_df

            # Force immediate refresh so all parts read new DF, switch to View
            st.session_state["switch_to_view"] = True
            st.rerun()
        except Exception as e:
            st.error(f"Validation/save failed: {e}")

    if reset_clicked:
        # Revert canonical DF to last saved JSON
        st.session_state["stores_table"] = load_from_json(
            stores_json_path, COLUMN_DTYPES
        )
        st.session_state["stores_editor_key"] = gen_randkey()
        st.rerun()

# -----------------------------
# VIEW BRANCH
# -----------------------------
else:
    st.title("Table View")
    table = st.session_state["stores_table"]

    table_toggle = st.columns([1])[0]
    with table_toggle:
        show_editor = st.toggle("Show editable table", value=True, key="show_editor")

    filter_toggle = st.columns([1])[0]
    with filter_toggle:
        if ("apply_filters" not in st.session_state) or (
            st.session_state["apply_filters"] is False
        ):
            st.toggle("Apply filters", key="apply_filters", value=False)
        else:
            st.toggle("Apply filters", key="apply_filters", value=True)

    enable_filters = st.session_state.get("apply_filters", False)

    st.dataframe(
        filter_dataframe(table, enable_filters),
        column_config=column_config,
        height=TABLE_HEIGHT,
    )
