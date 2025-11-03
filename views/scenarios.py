from uuid import uuid4
from db.db_operations import (
    copy_folder_to_another,
    delete_folder_with_data,
    upsert_delete,
)
from params.time_allocation import AVAILABLE_VISIT_FREQUENCIES
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from datetime import datetime
import json
import textwrap
from concurrent.futures import ThreadPoolExecutor

from db.scenarios import Role, Scenario
from utils.data_loaders import load_default_input_parameters
from utils.dataframe_utils import (
    batch_apply_schedule_adjustments,
    download_dataframe,
    filter_dataframe,
    gen_randkey,
    load_from_json,
    render_editable_table,
    sidebar_load_download_section_from_session_state,
    validate_and_align_columns,
)
from utils.date_utils import ensure_datetime

from utils.dialogs import engine_running_dialog, regionalization_dialog
from utils.week_distribution import week_distribution
from utils.fleet_optimization import fleet_optimization_all
from utils.models_common import map_workweek_days


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def _open_update_dialog():
    st.session_state["update_stores_metadata_confirm_dialog"] = True


def _close_update_dialog():
    st.session_state["update_stores_metadata_confirm_dialog"] = False


def all_scenarios_view(all_scenarios, scenario_folder):
    cols = st.columns([1, 1, 1])

    if "clicker" not in st.session_state:
        clicker = 0
        st.session_state["clicker"] = clicker
    else:
        clicker = st.session_state["clicker"]

    if "confirm_delete_scenario" not in st.session_state:
        st.session_state.confirm_delete_scenario = None

    with cols[0]:
        if st.button(
            "\\+ Create a new scenario", key="create_btn", use_container_width=False
        ):
            st.session_state.create_mode = True
            st.session_state.editing_scenario_id = None
            st.rerun()

    if not all_scenarios:
        st.info("No scenarios yet. Click **Create a new scenario** to get started.")
    else:
        st.header("All Scenarios")
        st.write("Select a scenario to view details:")

        all_scenarios = sorted(all_scenarios, key=lambda x: x.modified_at, reverse=True)
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
                        left_col, right_col, clone_col, delete_col = st.columns(
                            [1, 1, 1, 1]
                        )
                        with left_col:
                            if st.button("Open", key=f"open_{s.id}"):
                                st.session_state.selected_scenario = s
                                st.rerun()
                        with right_col:
                            if st.button("✏️ Edit", key=f"edit_{s.id}"):
                                st.session_state.editing_scenario_id = s.id
                                st.session_state.create_mode = False
                                st.session_state.clone_mode = False
                        with clone_col:
                            if st.button("📋 Clone", key=f"clone_{s.id}"):
                                # Create a new scenario draft based on the existing one
                                st.session_state.clone_mode = True
                                st.session_state.create_mode = False
                                st.session_state.editing_scenario_id = s.id
                        with delete_col:
                            if st.button("🗑️ Delete", key=f"delete_{s.id}"):
                                st.session_state.confirm_delete_scenario = s.id
                                # all_scenarios.remove(s)
                                # single_scenario_folder = scenario_folder / s.id
                                # if single_scenario_folder.exists():
                                #     shutil.rmtree(single_scenario_folder)
                                # st.success(f"Scenario '{s.name}' deleted.")
                                # st.rerun()
                        if st.session_state.confirm_delete_scenario:
                            if s.id == st.session_state.confirm_delete_scenario:
                                st.warning(
                                    f"Are you sure you want to delete **'{s.name}'**? This action cannot be undone."
                                )
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button(
                                        "✅ Yes, delete", key=f"confirm_yes_{s.id}"
                                    ):
                                        all_scenarios.remove(s)
                                        single_scenario_folder = scenario_folder / s.id
                                        delete_folder_with_data(single_scenario_folder)
                                        st.session_state.confirm_delete_scenario = None
                                        st.rerun()
                                with col2:
                                    if st.button("❌ Cancel", key=f"confirm_no_{s.id}"):
                                        st.session_state.confirm_delete_scenario = None
                                        st.info("Deletion cancelled.")
                                        st.rerun()

                        st.markdown("</div>", unsafe_allow_html=True)


# --- Sidebar form ---
def scenario_sidebar_form(
    project_id, scenario_folder, is_new=True, scenario_obj=None, is_clone=False
):
    if (
        st.session_state.create_mode
        or st.session_state.editing_scenario_id
        or st.session_state.clone_mode
    ):
        st.sidebar.markdown(f"### {'Create Scenario' if is_new else 'Edit Scenario'}")
        form_key = f"form_edit_{scenario_obj.id}" if scenario_obj else "form_create"
        if is_clone:
            original_scenario_id = scenario_obj.id
            scenario_obj.name = scenario_obj.name + " (Copy)"
        with st.sidebar.form(form_key, clear_on_submit=False):
            name = st.text_input(
                "Name", value=scenario_obj.name if scenario_obj else ""
            )
            description = st.text_area(
                "Description", value=scenario_obj.description if scenario_obj else ""
            )
            author = st.text_input(
                "Author", value=scenario_obj.author if scenario_obj else "admin"
            )
            # left_col_form, right_col_form = st.columns([1, 1])
            # with left_col_form:
            submitted = st.form_submit_button(
                "Save Scenario", type="primary", use_container_width=True
            )
            # with right_col_form:
            cancelled = st.form_submit_button("Cancel", use_container_width=True)

            if submitted:
                if is_new:
                    if (
                        scenario_folder.parent.parent / project_id / "stores.json"
                    ).exists():
                        new_scenario = Scenario(
                            project_id=project_id,
                            name=name,
                            description=description,
                            author=author,
                        )
                        new_scenario.save(scenario_folder)
                        new_scenario.save_initial_stores_w_role_spec(scenario_folder)
                        st.success("Scenario created successfully!")
                        st.session_state.create_mode = False
                        st.rerun()
                    else:
                        st.error(
                            "Please add stores to the project before creating a scenario."
                        )
                else:
                    # Edit existing scenario
                    scenario_obj.name = name
                    scenario_obj.description = description
                    scenario_obj.author = author
                    scenario_obj.modified_at = datetime.now()

                    if st.session_state.clone_mode:
                        print("cloning")
                        scenario_obj.id = str(uuid4())[:8]
                        print(scenario_folder / original_scenario_id)
                        print(scenario_folder / scenario_obj.id)
                        copy_folder_to_another(
                            scenario_folder / original_scenario_id,
                            scenario_folder / scenario_obj.id,
                        )

                    scenario_obj.save(scenario_folder)

                    st.success("Scenario updated successfully!")
                    st.session_state.editing_scenario_id = None
                    st.session_state.clone_mode = False
                    st.rerun()

            if cancelled:
                st.session_state.create_mode = False
                st.session_state.editing_scenario_id = None
                st.session_state.clone_mode = False
                st.rerun()


def time_allocation_res(scenario: Scenario, scenario_folder: Path):
    active_project = st.session_state.get("active_project")
    project_id = active_project["id"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    project_data_path = Path("data") / "projects" / project_id
    project_data_path.mkdir(parents=True, exist_ok=True)
    stores_json_path = project_data_path / "stores.json"
    scenario_json_path = (
        project_data_path / "scenarios" / scenario.id / "scenario_metadata.json"
    )
    stores_w_role_spec_json_path = (
        project_data_path / "scenarios" / scenario.id / "stores_w_role_spec.json"
    )
    project_params_path = project_data_path / "input_parameters.json"

    with open(project_params_path, "r", encoding="utf-8") as f:
        project_params = json.load(f)

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

    with open(stores_json_path, "r") as f:
        stores = pd.DataFrame(json.load(f))

    # Load scenario metadata and stores with role spec
    with open(scenario_json_path, "r") as f:
        scen_metadata = json.load(f)

    # Load stores with role spec
    with open(stores_w_role_spec_json_path, "r", encoding="utf-8") as f:
        stores_w_role_spec = pd.DataFrame(json.load(f))

    st.session_state["stores_w_role_spec"] = stores_w_role_spec

    if "stores_editor_key" not in st.session_state:
        st.session_state["stores_editor_key"] = gen_randkey()

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = "uploader_0"

    if "show_editor" not in st.session_state:
        st.session_state["show_editor"] = False

    COLUMN_DTYPES_TO_DOWNLOAD = COLUMN_DTYPES.copy()

    time_in_store_config_columns = {}
    for role in scen_metadata["roles"]:
        time_in_store_config_columns[f"{role['name']} - time in store"] = "float"
        COLUMN_DTYPES_TO_DOWNLOAD[f"{role['name']} - time in store"] = "float"

    time_in_store_columns_config = {
        col: st.column_config.Column(
            "\n".join(textwrap.wrap(col, width=10)),
            width="auto",
            disabled=False,
        )
        for col in time_in_store_config_columns.keys()
    }

    visits_config_columns = {}
    for role in scen_metadata["roles"]:
        visits_config_columns[f"{role['name']} - visits per month"] = "int"
        COLUMN_DTYPES_TO_DOWNLOAD[f"{role['name']} - visits per month"] = "int"

    visits_config_columns_config = {
        col: st.column_config.SelectboxColumn(
            "\n".join(textwrap.wrap(col, width=10)),
            width="auto",
            disabled=False,
            options=AVAILABLE_VISIT_FREQUENCIES,
        )
        for col in visits_config_columns.keys()
    }

    column_config = {
        col: st.column_config.Column(
            "\n".join(textwrap.wrap(col, width=10)), width="auto", disabled=True
        )
        for col in COLUMN_DTYPES.keys()
    }

    column_config = (
        column_config | time_in_store_columns_config | visits_config_columns_config
    )

    # -----------------------------
    # SIDEBAR LOAD / SAVE
    # -----------------------------
    with st.sidebar:
        sidebar_load_download_section_from_session_state(
            "stores_w_role_spec",
            stores_w_role_spec_json_path,
            project_id,
            "Customer",
            COLUMN_DTYPES_TO_DOWNLOAD,
        )

    # -----------------------------
    # EDITOR BRANCH
    # -----------------------------
    table = st.session_state["stores_w_role_spec"]
    stores_w_role_spec = table.copy()

    weekdays = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]
    workweek_days = map_workweek_days(project_params["week_schedule"])
    weekdays_to_drop = [day for day in weekdays if day not in workweek_days]
    for day in weekdays_to_drop:
        if day in table.columns:
            table = table.drop(columns=[day])

    filtered_table = filter_dataframe(df=table, enable_filters=True)

    if_adj_submitted, schedule_adjustments = batch_apply_schedule_adjustments(
        filtered_table, scen_metadata
    )

    if if_adj_submitted:
        stores_w_role_spec.loc[
            schedule_adjustments["indexes_to_modify"],
            f"{schedule_adjustments['selected_role']} - visits per month",
        ] = schedule_adjustments["visits_nb"]
        stores_w_role_spec.loc[
            schedule_adjustments["indexes_to_modify"],
            f"{schedule_adjustments['selected_role']} - time in store",
        ] = schedule_adjustments["duration"]
        with open(stores_w_role_spec_json_path, "w", encoding="utf-8") as f:
            json.dump(
                stores_w_role_spec.to_dict(orient="records"),
                f,
                ensure_ascii=False,
                indent=2,
            )
        st.rerun()

    stores_w_role_spec_filt = stores_w_role_spec.loc[
        schedule_adjustments["indexes_to_modify"]
    ]

    stores_w_role_spec_filt = filtered_table

    input_parameter_dict = load_default_input_parameters(project_id)

    cols_to_hide = [
        "Latitude",
        "Longitude",
    ]

    weekdays = [day for day in weekdays if day in stores_w_role_spec_filt.columns]

    order_of_the_columns = [
        col for col in stores_w_role_spec_filt.columns if col not in weekdays
    ] + weekdays

    stores_w_role_spec_filt = stores_w_role_spec_filt[order_of_the_columns]

    if not input_parameter_dict["consider_delivery"]:
        cols_to_hide += [
            day for day in weekdays if day in stores_w_role_spec_filt.columns
        ]

    ## EDITABLE TABLE
    render_editable_table(
        stores_w_role_spec_filt,
        column_config,
        "Customer",
        stores_w_role_spec_json_path,
        "time_allocation",
        cols_to_hide,
    )

    # --- Init state once
    if "update_stores_metadata_confirm_dialog" not in st.session_state:
        st.session_state["update_stores_metadata_confirm_dialog"] = False

    # --- Primary button just toggles the dialog flag (DO NOT assign from st.button return)
    st.button(
        "Update stores metadata",
        key="open_update_stores_metadata",
        on_click=_open_update_dialog,
    )

    if st.session_state["update_stores_metadata_confirm_dialog"]:
        c1, c2, _ = st.columns([1, 1, 3])

        with c1:
            if st.button(
                "✅ Yes, update",
                key="yes_update_stores_metadata",
                use_container_width=True,
            ):
                visits_data_columns = [
                    c
                    for c in stores_w_role_spec.columns
                    if (c == "Customer")
                    or (" - visits per month" in c)
                    or ("- time in store" in c)
                ]
                visits_data = stores_w_role_spec[visits_data_columns]
                stores_w_role_spec_mod = pd.merge(
                    stores,
                    visits_data,
                    how="left",
                    on="Customer",
                    validate="1:1",
                )
                with open(stores_w_role_spec_json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        stores_w_role_spec_mod.to_dict(orient="records"),
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                scenario.regionalization_completed = False
                scenario.week_distribution_completed = False
                scenario.route_optimization_completed = False
                scenario.save(scenario_folder)

                for role in scenario.roles:
                    scenario.remove_role_results(scenario_folder, role.name)

                _close_update_dialog()
                st.rerun()

        with c2:
            if st.button(
                "❌ Cancel",
                key="cancel_update_stores_metadata",
                use_container_width=True,
            ):
                st.info("Update cancelled.")
                _close_update_dialog()
                st.rerun()


def regionalization_res():
    st.subheader("Map Visualization")
    map_data = pd.DataFrame(
        np.random.randn(10, 2) / [50, 50] + [37.77, -122.42], columns=["lat", "lon"]
    )
    st.map(map_data)


def fleet_opt_res():
    st.subheader("Chart 2")
    st.bar_chart(np.random.randn(10, 4))


def metric_card(s: Scenario, title: str, key: str = "metric_card"):
    # Render slider first (Streamlit input must be separate from HTML)
    value = st.slider(
        label=f"{title} ",
        min_value=0,
        max_value=100,
        value=s.time_in_store,
        step=1,
        key=f"{key}_slider",
        label_visibility="collapsed",
    )
    return value


def new_metadata_section(project_id, scenario_folder):
    s = st.session_state.selected_scenario

    input_parameter_dict = load_default_input_parameters(project_id)

    working_hours = input_parameter_dict.get("working_hours", None)
    # --------------------------
    # Metadata Section Layout
    # --------------------------
    left_col, right_col = st.columns([2, 1])

    # --------------------------
    # LEFT COLUMN
    # --------------------------
    with left_col:
        created_at = ensure_datetime(s.created_at)
        modified_at = ensure_datetime(s.modified_at)

        metadata_items = {
            "Name": s.name,
            "Description": s.description or "-",
            "Author": s.author,
            "ID": s.id,
            "Created": created_at.strftime("%Y-%m-%d %H:%M"),
            "Modified": modified_at.strftime("%Y-%m-%d %H:%M"),
            "Working hours in month": f"{working_hours}h" if working_hours else "-",
            "Time in store in a month": (
                f"{int(round(s.time_in_store*working_hours/100,0))}h"
                if (s.time_in_store and working_hours)
                else "-"
            ),
        }

        with st.container(border=True):  # native border and padding
            left_cols = st.columns([3, 2])

            with left_cols[0]:
                for key, value in metadata_items.items():
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.markdown(f"**{key}:**")
                    with col2:
                        st.write(value if value else "-")

            with left_cols[1]:
                st.write(f"### Time in store factor")
                time_in_store_desc = f"""<b>Time in store factor</b> measures how much time in store, single FTE can spend across all working hours in a month. 
                This parameter will be <b>automatically estimated</b> before 
                    Regionalization whenever its value is set as 100%, or <b>it can be manually adjusted</b>. Lower value means more time spent on the road, higher value means more time in store.
                    Its value depends on region's characteristics."""
                st.markdown(f"<p>ℹ️ {time_in_store_desc}</p>", unsafe_allow_html=True)
                updated_value = metric_card(
                    s,
                    title="Time in store",
                    key="time_in_store_card",
                )

                if updated_value != s.time_in_store:
                    s.time_in_store = updated_value
                    s.save(scenario_folder)
                    st.success("Time in store updated successfully!")
                    st.rerun()

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
                            if hasattr(role, "visits_window") and role.visits_window:
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
                                            <b>Visits Window vs Delivery Dates:</b> {role.visits_window[0]}h to {role.visits_window[1]}h
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            else:
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
                                            s.remove_role_results(
                                                scenario_folder, role.name
                                            )
                                            s.update_process_step_status(
                                                scenario_folder
                                            )
                                            s.save(scenario_folder)
                                            s.remove_cols_from_stores_w_role_spec(
                                                scenario_folder, role.name
                                            )
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
                        disabled=False if not editing_role else True,
                    )
                    description = st.text_area(
                        "Description",
                        value=editing_role.description if editing_role else "",
                    )

                    # --- Defaults ---
                    project_data_path = Path("data") / "projects" / project_id
                    stores_json_path = project_data_path / "stores.json"
                    base = load_from_json(
                        stores_json_path, config["stores_input_dtypes"]
                    )
                    st.session_state["stores_table"] = validate_and_align_columns(
                        base, config["stores_input_dtypes"]
                    )
                    role_spec_input = (
                        st.session_state["stores_table"][["Group", "Grading"]]
                        .drop_duplicates()
                        .sort_values(["Group", "Grading"])
                    )
                    role_spec_input[["Visits per month", "Time in store"]] = 0

                    if not editing_role:
                        st.markdown("**Visits number and visiting time per month**")
                        role_spec_filled = st.data_editor(
                            role_spec_input,
                            key="editable_table",
                            hide_index=True,
                            column_config={
                                c: st.column_config.TextColumn(disabled=True)
                                for c in ["Group", "Grading"]
                            }
                            | {
                                "Visits per month": st.column_config.SelectboxColumn(
                                    "\n".join(
                                        textwrap.wrap("Visits per month", width=10)
                                    ),
                                    width="auto",
                                    help="Select visits per month",
                                    options=AVAILABLE_VISIT_FREQUENCIES,
                                    disabled=False,
                                    default=0,
                                )
                            },
                        )

                        role_spec_filled = role_spec_filled.rename(
                            columns={
                                "Visits per month": f"{name} - visits per month",
                                "Time in store": f"{name} - time in store",
                            }
                        )

                    # --- Visits Window ---
                    input_parameter_dict = load_default_input_parameters(project_id)

                    if input_parameter_dict["consider_delivery"]:
                        st.markdown("**Visits Window (relative to Delivery Day)**")
                        visits_window = st.slider(
                            "Select allowed visit window (hours relative to Delivery Day)",
                            min_value=-72,
                            max_value=72,
                            value=(
                                editing_role.visits_window if editing_role else (0, 24)
                            ),
                            step=24,
                            format="%d h",
                            help="0h = Delivery Day, +24h = next day, -24h = previous day",
                        )
                        st.caption("📍 0h = Delivery Day (reference point)")
                    else:
                        visits_window = None

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
                        if (name in [r.name for r in roles]) and (not editing_role):
                            st.warning(
                                f"Role '{name}' already exists. Please choose another name."
                            )
                        elif (not name) or (name == ""):
                            st.warning(f"Role's name cannot be empty.")
                        else:
                            if editing_role:
                                # Update existing role
                                editing_role.name = name
                                editing_role.description = description
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
                                    role_spec_filled=role_spec_filled.to_dict(
                                        orient="records"
                                    ),
                                    visits_window=visits_window,
                                )
                                s.add_role(role)
                                s.update_stores_w_role_spec(scenario_folder, role)
                                st.success("Role created successfully!")

                            s.save(scenario_folder)
                            st.session_state.role_expander_open = False
                            st.rerun()

    # --------------------------
    # RIGHT COLUMN
    # --------------------------
    with right_col:
        status_mapping = {
            "1. Regionalization": {
                "completed": s.regionalization_completed,
                "info": f"""This step is <b>estimating the number of FTEs</b>, for each specified role, required to cover all the 
                visits in the scenario, specified in <b>Time Allocation</b> input table. Maximum capacity per FTE is using <b>Working hours in a month</b> 
                parameter, multiplied by <b>Time in store</b> factor. <b>Time in store</b> factor is estimated on a run of Regionalization engine if value is equal to 100%, or can be manually selected.""",
            },
            "2. Week Distribution": {
                "completed": s.week_distribution_completed,
                "info": "Second engine which spreads visits across the month for each role defined, according to visit frequencies and time in store",
            },
            "3. Route Optimization": {
                "completed": s.route_optimization_completed,
                "info": "Third engine, utilizing Google Maps API, which determines the most efficient sequence of site visits for role's assigned territory and visits across the month",
            },
        }

        # ---- Init state ----
        if "show_confirm_dialog" not in st.session_state:
            st.session_state.show_confirm_dialog = False
        if "selected_step" not in st.session_state:
            st.session_state.selected_step = None
        # if "engine_running" not in st.session_state:
        #     st.session_state.engine_running = False

        # ---- Handlers ----
        def confirm_run(step):
            st.session_state.show_confirm_dialog = True
            st.session_state.selected_step = step
            st.session_state.force_rerun = True

        executor = ThreadPoolExecutor(max_workers=4)

        def run_model(step, scenario, session_state_key="engine_running"):
            if step == "1. Regionalization":
                st.session_state["show_regionalization_dialog"] = True
                st.rerun()
            elif step == "2. Week Distribution":
                engine_running_dialog(
                    step,
                    task_fn=lambda: executor.submit(
                        week_distribution(scenario_folder, scenario)
                    ),
                    session_state_key=session_state_key,
                )
            elif step == "3. Route Optimization":
                engine_running_dialog(
                    step,
                    task_fn=lambda: executor.submit(
                        fleet_optimization_all(scenario_folder, scenario)
                    ),
                    session_state_key=session_state_key,
                )

            # s.save(scenario_folder)
            st.session_state.show_confirm_dialog = False
            st.session_state.selected_step = None
            st.session_state.force_rerun = True

        def cancel_run():
            st.session_state.show_confirm_dialog = False
            st.session_state.selected_step = None
            st.session_state.force_rerun = True

        if st.session_state.get("show_regionalization_dialog", False):
            regionalization_dialog(scenario_folder, s)
            st.session_state.show_confirm_dialog = False
            st.session_state.selected_step = None
            st.session_state.force_rerun = True

        # ---- Layout ----
        for key, info in status_mapping.items():
            with st.container(border=True):
                st.write(f"### {key}")
                st.markdown(f"<p>ℹ️ {info['info']}</p>", unsafe_allow_html=True)

                if key == "3. Route Optimization":
                    cost_estimation = s.estimate_route_optimization(scenario_folder)
                    if cost_estimation is not None:
                        cost_estimation_adj = (
                            cost_estimation * config["g_routing_cost_est_factor"]
                        )
                        st.info(
                            f"Estimated cost for this run: {int(np.round(cost_estimation_adj, 0))} USD."
                        )

                st.write("✅ Completed" if info["completed"] else "❌ Not completed")

                btn_label = (
                    "Rerun the engine" if info["completed"] else "Run the engine"
                )
                col1, col2 = st.columns([3, 2])
                with col2:
                    st.button(
                        btn_label,
                        key=f"btn_{key}",
                        on_click=confirm_run,
                        args=(key,),
                    )

                # ---- Confirmation dialog under the clicked card ----
                if (
                    st.session_state.show_confirm_dialog
                    and st.session_state.selected_step == key
                ):
                    st.warning(
                        f"Are you sure you want to run **{key}**?\n\n"
                        "⚠️ This will overwrite existing results."
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(
                            "✅ Yes, run",
                            key=f"confirm_yes_{key}",
                            # on_click=run_model,
                            # args=(key, s.id),
                        ):
                            run_model(key, s)
                    with col2:
                        st.button(
                            "❌ Cancel", key=f"confirm_no_{key}", on_click=cancel_run
                        )
