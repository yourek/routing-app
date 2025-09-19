from db.db_operations import upsert_delete
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from session.session import init_session
from utils.auth_utils.guards import require_active_project
from db.scenarios import Role, Scenario
from utils.dataframe_utils import (
    filter_dataframe,
    gen_randkey,
    get_deleted_rows_keys,
    load_from_json,
    to_excel_bytes,
    validate_and_align_columns,
    validate_new_row_addition,
)
from utils.date_utils import ensure_datetime


init_session()
require_active_project()

st.title("🧩 Add & Manage Scenarios")


active_project = st.session_state.get("active_project")
project_name = active_project["name"]
project_id = active_project["id"]

# Create project data folder if not created
SCENARIO_FOLDER = Path("data") / "projects_data" / project_id / "scenarios"
SCENARIO_FOLDER.mkdir(parents=True, exist_ok=True)

# SCENARIO_FOLDER = Path("data") / "projects_data" / project_id / "scenarios"
# SCENARIO_FOLDER.mkdir(parents=True, exist_ok=True)

# # Create a scenario
# scenario = Scenario(project_id="project_123", name="Scenario 1", author="Alice")
# scenario.add_role(Role("Planner"))
# scenario.add_role(Role("Merchandiser"))

# # st.write(scenario.to_dict())
# # Save to JSON
# scenario.save(SCENARIO_FOLDER)

# Load all scenarios
all_scenarios = Scenario.load_all(SCENARIO_FOLDER)
# for s in all_scenarios:
#     print(s.name, [r.name for r in s.roles])

# st.write(f"Loaded {len(all_scenarios)} scenarios from disk.")
# st.write([s.to_dict() for s in all_scenarios])


def new_metadata_section():
    s = st.session_state.selected_scenario  # your Scenario object

    # --------------------------
    # Metadata Section Layout
    # --------------------------
    left_col, right_col = st.columns([2, 1])  # left wider than right

    # --------------------------
    # LEFT COLUMN
    # --------------------------
    with left_col:
        # Top container: name + description
        # st.markdown(
        #     f"""
        #     <div style="background-color:#f0f2f6; padding:15px; border-radius:8px; margin-bottom:10px;">
        #         <h4 style="margin:0;">Name</h4>
        #         <p style="margin:2px 0 5px 0;"><b>Name:</b> {s.name}</p>
        #         <h4 style="margin:0;">Description</h4>
        #         <p style="margin:2px 0 0 0;">{s.description or '-'}</p>
        #     </div>
        #     """,
        #     unsafe_allow_html=True,
        # )
        created_at = ensure_datetime(s.created_at)
        modified_at = ensure_datetime(s.modified_at)

        metadata_items = {
            "Name": s.name,
            "Description": s.description or "-",
            "Author": s.author,
            "ID": s.id,
            "Created": created_at.strftime("%Y-%m-%d %H:%M"),
            "Modified": modified_at.strftime("%Y-%m-%d %H:%M"),
        }

        meta_html = ""
        for key, value in metadata_items.items():
            meta_html += f"""<div style='display: flex; align-items: center; gap: 10px; padding: 10px; background-color: #f0f2f6;'>
                <h5 style='margin:0; padding: 0; min-width: 120px;'>{key}:</h5>
                <p style='margin:0;'>{value}</p></div>
                """

        st.markdown(
            f"""
            <div style="background-color:#f0f2f6; padding:15px; border-radius:8px; margin-bottom:10px;">
                {meta_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.subheader("Roles within a scenario")

            roles = s.roles
            if roles:
                chunk_size = 3  # max cards per row
                for i in range(0, len(roles), chunk_size):
                    row_roles = roles[i : i + chunk_size]
                    cols = st.columns(len(row_roles))
                    for col, role in zip(cols, row_roles):
                        with col:
                            # Card with name + description + buttons
                            st.markdown(
                                f"""
                                <div style="
                                    border: 1px solid #ccc; 
                                    border-radius: 8px; 
                                    padding: 12px; 
                                    background-color: #f9f9f9;
                                    min-height: 120px;
                                    display: flex;
                                    flex-direction: column;
                                    justify-content: space-between;
                                ">
                                    <h4 style="margin: 0;">{role.name}</h4>
                                    <p style="margin: 4px 0 0 0; font-size: 0.9em; color: #555;"><b>Description:</b> {role.description or '-'}</p>
                                    <p style="margin: 8px 0 0 0; font-size: 0.85em; color: #333;">
                                        <b>Default Visits/Month:</b> {role.default_visits}<br>
                                        <b>Default Hours/Month:</b> {role.default_hours}
                                    </p>
                                    <p style="margin: 8px 0 0 0; font-size: 0.85em; color: #333;">
                                        <b>Visits Window vs Delivery Dates:</b> {role.visits_window[0]}h to {role.visits_window[1]}h
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # Buttons below card
                            edit_col, delete_col = st.columns([1, 1])
                            with edit_col:
                                if st.button("✏️ Edit", key=f"edit_{role.id}"):
                                    st.session_state.editing_role_id = role.id
                                    st.rerun()
                            with delete_col:
                                # Check if this role is pending delete
                                pending_delete = (
                                    st.session_state.get("pending_delete_role_id")
                                    == role.id
                                )

                                if pending_delete:
                                    st.warning(f"⚠️ Confirm deletion of '{role.name}'?")
                                    confirm_col, cancel_col = st.columns([1, 1])
                                    with confirm_col:
                                        if st.button(
                                            "✅ Yes", key=f"confirm_{role.id}"
                                        ):
                                            s.roles = [
                                                r for r in s.roles if r.id != role.id
                                            ]
                                            s.save(SCENARIO_FOLDER)
                                            st.session_state.pop(
                                                "pending_delete_role_id", None
                                            )
                                            st.rerun()
                                    with cancel_col:
                                        if st.button("❌ No", key=f"cancel_{role.id}"):
                                            st.session_state.pop(
                                                "pending_delete_role_id", None
                                            )
                                            st.rerun()
                                else:
                                    if st.button("🗑️ Delete", key=f"delete_{role.id}"):
                                        st.session_state["pending_delete_role_id"] = (
                                            role.id
                                        )
                                        st.rerun()

            else:
                st.info("No roles created yet.")

            if "role_expander_open" not in st.session_state:
                st.session_state.role_expander_open = False

            editing_role = None
            if "editing_role_id" in st.session_state:
                editing_role = next(
                    (r for r in s.roles if r.id == st.session_state.editing_role_id),
                    None,
                )
                st.session_state.role_expander_open = True  # expand if editing
            elif not st.session_state.role_expander_open:
                st.session_state.role_expander_open = False  # collapsed for new

            # with st.expander("➕ Create New Role", expanded=False):
            #     with st.form("new_role_form", clear_on_submit=True):
            #         st.markdown("Define New Role")

            #         # --- Basic info ---
            #         name = st.text_input("Role Name")
            #         description = st.text_area("Description")

            #         # --- Defaults ---
            #         default_visits = st.number_input(
            #             "Default visits per month", min_value=0, step=1
            #         )
            #         default_hours = st.number_input(
            #             "Default hours per month", min_value=0.0, step=0.5
            #         )

            #         # --- Visits Window ---
            #         st.markdown("**Visits Window (relative to Delivery Day)**")

            #         # Slider range: -72h ... +72h (3 days before/after)
            #         # step = 24h = 1 tick
            #         visits_window = st.slider(
            #             "Select allowed visit window (hours relative to Delivery Day)",
            #             min_value=-72,
            #             max_value=72,
            #             value=(0, 24),  # default: delivery day to +24h
            #             step=24,
            #             format="%d h",
            #             help="0h = Delivery Day, +24h = next day, -24h = previous day",
            #         )

            #         # Delivery Day reference marker
            #         st.caption("📍 0h = Delivery Day (reference point)")

            #         # --- Submit button ---
            #         submitted = st.form_submit_button("✅ Save Role")

            #         if submitted:
            #             role = Role(
            #                 project_id=project_id,
            #                 scenario_id=s.id,
            #                 name=name,
            #                 description=description,
            #                 default_visits=default_visits,
            #                 default_hours=default_hours,
            #                 visits_window=visits_window,
            #             )
            #             s.add_role(role)
            #             s.save(SCENARIO_FOLDER)

            #             st.success("Role created successfully!")
            #             st.json(
            #                 {
            #                     "name": name,
            #                     "description": description,
            #                     "default_visits": default_visits,
            #                     "default_hours": default_hours,
            #                     "visits_window": visits_window,
            #                 }
            #             )
            #             st.rerun()

            # # Expander: expand if editing, otherwise collapsed
            expander_label = (
                "➕ Create New Role"
                if not editing_role
                else f"✏️ Edit Role: {editing_role.name}"
            )
            with st.expander(
                expander_label, expanded=st.session_state.role_expander_open
            ):
                with st.form("new_role_form", clear_on_submit=True):
                    st.markdown(expander_label)

                    # --- Basic info ---
                    name = st.text_input(
                        "Role Name",
                        value=editing_role.name if editing_role else "",
                    )
                    description = st.text_area(
                        "Description",
                        value=editing_role.description if editing_role else "",
                    )

                    # --- Defaults ---
                    default_visits = st.number_input(
                        "Default visits per month",
                        min_value=0,
                        step=1,
                        value=editing_role.default_visits if editing_role else 0,
                    )
                    default_hours = st.number_input(
                        "Default hours per month",
                        min_value=0.0,
                        step=0.5,
                        value=editing_role.default_hours if editing_role else 0.0,
                    )

                    # --- Visits Window ---
                    st.markdown("**Visits Window (relative to Delivery Day)**")
                    visits_window = st.slider(
                        "Select allowed visit window (hours relative to Delivery Day)",
                        min_value=-72,
                        max_value=72,
                        value=editing_role.visits_window if editing_role else (0, 24),
                        step=24,
                        format="%d h",
                        help="0h = Delivery Day, +24h = next day, -24h = previous day",
                    )
                    st.caption("📍 0h = Delivery Day (reference point)")

                    # --- Buttons button ---
                    edit_col, delete_col, rest_col = st.columns([1, 1, 2])
                    with edit_col:
                        submitted = st.form_submit_button("✅ Save Role")
                    with delete_col:
                        cancelled = st.form_submit_button("❌ Cancel")

                    if cancelled:
                        if editing_role:
                            st.session_state.pop("editing_role_id")
                            st.rerun()

                    if submitted:
                        if editing_role:
                            # Update existing role
                            editing_role.name = name
                            editing_role.description = description
                            editing_role.default_visits = default_visits
                            editing_role.default_hours = default_hours
                            editing_role.visits_window = visits_window

                            st.session_state.pop("editing_role_id")
                            st.success(f"Role '{name}' updated successfully!")
                        else:
                            # Create new role
                            role = Role(
                                project_id=project_id,
                                scenario_id=s.id,
                                name=name,
                                description=description,
                                default_visits=default_visits,
                                default_hours=default_hours,
                                visits_window=visits_window,
                            )
                            s.add_role(role)
                            st.success("Role created successfully!")

                        # Save scenario and rerun to refresh UI
                        s.save(SCENARIO_FOLDER)
                        st.session_state.role_expander_open = False
                        st.rerun()
    # --------------------------
    # RIGHT COLUMN
    # --------------------------
    with right_col:
        status_mapping = {
            "Regionalization": s.regionalization_completed,
            "Week Distribution": s.week_distribution_completed,
            "Route Optimization": s.route_optimization_completed,
        }

        for key, completed in status_mapping.items():
            icon = "✅" if completed else "❌"
            color = "green" if completed else "gray"
            st.markdown(
                f"""
                <div style="background-color:#f0f2f6; padding:15px; border-radius:8px; margin-bottom:10px; text-align:center;">
                    <h4 style="margin:0;">{key}</h4>
                    <p style="font-size:24px; margin:5px 0 0 0; color:{color};">{icon}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------
# Session state for selected scenario
# --------------------------
if "selected_scenario" not in st.session_state:
    st.session_state.selected_scenario = None


# --------------------------
# Default view: all scenarios
# --------------------------
if st.session_state.selected_scenario is None:
    cols = st.columns([1, 1, 1])

    if "clicker" not in st.session_state:
        clicker = 0
        st.session_state["clicker"] = clicker
    else:
        clicker = st.session_state["clicker"]

    with cols[0]:
        if st.button(
            "\\+ Create a new scenario", key="create_btn", use_container_width=False
        ):
            # Create a scenario
            description = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
            scenario = Scenario(
                project_id=project_id,
                name=f"Scenario {clicker + 1}",
                author="admin",
                description=description,
            )
            if clicker == 0:
                scenario.add_role(Role("Sales Representative"))
                scenario.add_role(Role("Merchandiser"))
            elif clicker == 1:
                scenario.add_role(Role("Junior Sales Representative"))
                scenario.add_role(Role("Senior Sales Representative"))
                scenario.add_role(Role("Merchandiser"))
            elif clicker == 2:
                scenario.add_role(Role("Junior Sales Representative"))
                scenario.add_role(Role("Senior Sales Representative"))
                scenario.add_role(Role("Junior Merchandiser"))
                scenario.add_role(Role("Senior Merchandiser"))

            # st.write(scenario.to_dict())
            # Save to JSON
            scenario.save(SCENARIO_FOLDER)
            clicker += 1
            st.session_state["clicker"] = clicker
            st.rerun()
            # # Prepare an unsaved draft and open the edit dialog
            # draft = new_project_draft(author=USER)
            # st.session_state["edit_id"] = draft["id"]
            # # holds values while editing (unsaved)
            # st.session_state["edit_data"] = draft
            # st.session_state["is_new"] = True
            # # no rerun: the edit sidebar below will render this run
    if not all_scenarios:
        st.info("No scenarios yet. Click **Create a new scenario** to get started.")
    else:
        st.header("All Scenarios")
        st.write("Select a scenario to view details:")

        # for s in all_scenarios:
        #     cols = st.columns([6, 1])
        #     cols[0].write(f"**{s.name}**")
        #     if cols[1].button("Open", key=f"open_{s.id}"):
        #         st.session_state.selected_scenario = s
        #         st.rerun()

        n_per_row = 3
        rows = [
            all_scenarios[i : i + n_per_row]
            for i in range(0, len(all_scenarios), n_per_row)
        ]
        for row in rows:
            cols = st.columns(n_per_row, gap="large")
            for col, s in zip(cols, row):
                with col:
                    with st.container(border=True):
                        st.markdown(
                            '<div class="scenario-card">',
                            unsafe_allow_html=True,
                        )

                        # Top: name and active badge
                        name_line = f"### {s.name}"

                        st.markdown(name_line, unsafe_allow_html=True)

                        # Middle: fields
                        st.markdown(
                            f"""
                            <div class="scenario-fields">
                            <div>{s.description or '<no description>'}</div>
                            <div style="opacity:.7;margin-top:8px;">author: {s.author}</div>
                            <div style="opacity:.7;margin-top:8px;">created at: {s.created_at}</div>
                            <div style="opacity:.7;margin-top:8px; margin-bottom:16px;">last modified at: {s.modified_at}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Bottom: Open button
                        if st.button("Open", key=f"open_{s.id}"):
                            st.session_state.selected_scenario = s
                            st.rerun()

                        st.markdown("</div>", unsafe_allow_html=True)


else:
    # --------------------------
    # Single scenario dashboard
    # --------------------------
    s = st.session_state.selected_scenario

    st.button(
        "⬅ Back to All Scenarios",
        on_click=lambda: st.session_state.update({"selected_scenario": None}),
    )
    st.header(f"Dashboard: {s.name}")

    # --------------------------
    # Top metadata section (cards)
    # --------------------------
    new_metadata_section()

    # st.subheader("Scenario Metadata")

    # # Map the scenario object fields to displayable metadata
    # meta = {
    #     "Project ID": s.project_id,
    #     "Author": s.author,
    #     "Created": s.created_at,
    #     "Modified": s.modified_at,
    #     "Roles": ", ".join([r.name for r in s.roles]),
    #     "Regionalization Done": s.regionalization_completed,
    #     "Week Distribution Done": s.week_distribution_completed,
    #     "Route Optimization Done": s.route_optimization_completed,
    # }

    # # Create cards (2 rows if needed)
    # cols_per_row = 4
    # items = list(meta.items())
    # for row_start in range(0, len(items), cols_per_row):
    #     cols = st.columns(min(cols_per_row, len(items) - row_start))
    #     for i, (key, value) in enumerate(items[row_start : row_start + cols_per_row]):
    #         with cols[i]:
    #             st.markdown(
    #                 f"""
    #                 <div style="background-color:#f0f2f6; padding:15px; border-radius:8px; text-align:center;">
    #                     <h4 style="margin:0;">{key}</h4>
    #                     <p style="font-size:16px; margin:5px 0 0 0;">{value}</p>
    #                 </div>
    #                 """,
    #                 unsafe_allow_html=True,
    #             )

    # st.markdown("---")  # horizontal divider

    # --------------------------
    # Bottom tabs
    # --------------------------
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Time Allocation",
            "Regionalization Results",
            "Week Distribution Results",
            "Fleet Optimization Results",
        ]
    )

    with tab1:

        active_project = st.session_state.get("active_project")
        project_name = active_project["name"]
        project_id = active_project["id"]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        project_data_path = Path("data") / "projects_data" / project_id
        project_data_path.mkdir(parents=True, exist_ok=True)
        stores_json_path = project_data_path / "stores.json"

        COLUMN_DTYPES = {
            "Customer": "str",
            "Stores": "str",
            "Chain": "str",
            "City": "str",
            "Street": "str",
            "Segment / Group": "str",
            "Latitude": "float",
            "Longitude": "float",
            "Sunday": "bool",
            "Monday": "bool",
            "Tuesday": "bool",
            "Wednesday": "bool",
            "Thursday": "bool",
            "Sales Rep - Visit Frequency": "int",
            "Sales Rep - Monthly Hours": "int",
            "Merchandiser - Visit Frequency": "int",
            "Merchandiser - Monthly Hours": "int",
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
            st.session_state["stores_table"] = validate_and_align_columns(
                base, COLUMN_DTYPES
            )

        if "stores_editor_key" not in st.session_state:
            st.session_state["stores_editor_key"] = gen_randkey()

        if "uploader_key" not in st.session_state:
            st.session_state["uploader_key"] = "uploader_0"

        if "show_editor" not in st.session_state:
            st.session_state["show_editor"] = False

        # Handle switch to Table View
        if st.session_state.get("switch_to_view"):
            st.session_state["show_editor"] = (
                False  # set widget value pre-instantiation
            )
            st.session_state.pop("switch_to_view", None)  # clear the control flag
            st.rerun()  # ensure the next run renders the else branch

        column_config = {
            col: st.column_config.Column(col, width="auto")
            for col in COLUMN_DTYPES.keys()
        }

        # -----------------------------
        # SIDEBAR LOAD / SAVE
        # -----------------------------
        with st.sidebar:
            st.subheader("Load / Save")

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
                    df_up = pd.read_excel(
                        uploaded_file,
                        sheet_name="Data",
                        engine="openpyxl",
                        dtype=COLUMN_DTYPES,
                    )
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
                    st.session_state["uploader_key"] = (
                        f"uploader_{datetime.now().timestamp()}"
                    )
                except Exception as e:
                    st.error(f"Error reading XLSX: {e}")

            # Reset the 'processed' flag when page shows with no file selected
            if uploaded_file is None and st.session_state.get(
                "upload_processed", False
            ):
                st.session_state["upload_processed"] = (
                    False  # ready for next upload [16]
                )

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
            # Feed editor from canonical DF directly to avoid divergence
            table = st.session_state["stores_table"].copy()

            # Buttons
            # table_toggle = st.columns([1])[0]
            # with table_toggle:
            #     show_editor = st.toggle(
            #         "Show editable table", value=True, key="show_editor"
            #     )

            # filter_toggle, c_save, c_reset, spacer = st.columns(
            #     [1, 1, 1, 7], gap="small"
            # )
            # with filter_toggle:
            #     if ("apply_filters" not in st.session_state) or (
            #         st.session_state["apply_filters"] is False
            #     ):
            #         st.toggle("Apply filters", key="apply_filters", value=False)
            #     else:
            #         st.toggle("Apply filters", key="apply_filters", value=True)
            # with c_save:
            #     save_clicked = st.button("Save changes", type="primary")
            # with c_reset:
            #     reset_clicked = st.button("Reset changes")

            # enable_filters = st.session_state.get("apply_filters", False)

            # edited = st.data_editor(
            #     filter_dataframe(table, enable_filters),
            #     column_config=column_config,
            #     num_rows="dynamic",
            #     key=st.session_state["stores_editor_key"],
            #     height=TABLE_HEIGHT,
            # )

            # if save_clicked:
            #     try:
            #         changed = edited.copy()

            #         validate_new_row_addition(changed, ID_COL, COLUMN_DTYPES)

            #         deleted_keys = get_deleted_rows_keys(
            #             table, ID_COL, st.session_state["stores_editor_key"]
            #         )
            #         result_df = upsert_delete(
            #             edited_df=changed,
            #             json_path=stores_json_path,
            #             key_col=ID_COL,
            #             deleted_keys=deleted_keys,
            #         )
            #         result_df = validate_and_align_columns(result_df, COLUMN_DTYPES)

            #         # Update canonical DF
            #         st.session_state["stores_table"] = result_df

            #         # Force immediate refresh so all parts read new DF, switch to View
            #         st.session_state["switch_to_view"] = True
            #         st.rerun()
            #     except Exception as e:
            #         st.error(f"Validation/save failed: {e}")

            # if reset_clicked:
            #     # Revert canonical DF to last saved JSON
            #     st.session_state["stores_table"] = load_from_json(
            #         stores_json_path, COLUMN_DTYPES
            #     )
            #     st.session_state["stores_editor_key"] = gen_randkey()
            #     st.rerun()

        # -----------------------------
        # VIEW BRANCH
        # -----------------------------
        else:
            table = st.session_state["stores_table"]

            # table_toggle = st.columns([1])[0]
            # with table_toggle:
            #     show_editor = st.toggle(
            #         "Show editable table", value=True, key="show_editor"
            #     )

            # filter_toggle = st.columns([1])[0]
            # with filter_toggle:
            #     if ("apply_filters" not in st.session_state) or (
            #         st.session_state["apply_filters"] is False
            #     ):
            #         st.toggle("Apply filters", key="apply_filters", value=False)
            #     else:
            #         st.toggle("Apply filters", key="apply_filters", value=True)

            # enable_filters = st.session_state.get("apply_filters", False)

            st.dataframe(
                filter_dataframe(table, True),
                column_config=column_config,
                height=TABLE_HEIGHT,
            )

    with tab2:
        st.subheader("Map Visualization")
        map_data = pd.DataFrame(
            np.random.randn(10, 2) / [50, 50] + [37.77, -122.42], columns=["lat", "lon"]
        )
        st.map(map_data)

    with tab3:
        st.subheader("Chart 1")
        chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["A", "B", "C"])
        st.line_chart(chart_data)

    with tab4:
        st.subheader("Chart 2")
        st.bar_chart(np.random.randn(10, 4))
