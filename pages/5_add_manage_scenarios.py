import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from session.session import init_session
from utils.auth_utils.guards import require_active_project, require_authentication
from db.scenarios import Role, Scenario

from views.scenarios import (
    all_scenarios_view,
    scenario_sidebar_form,
    time_allocation_res,
    new_metadata_section,
)

from views.scenarios_.regionalization_results import regionalization_results
from views.scenarios_.week_distr_results import week_distribution_res
from views.scenarios_.fleet_opt_results import fleet_opt_results


init_session()
require_authentication()
require_active_project()

st.title("🧩 Add & Manage Scenarios")


active_project = st.session_state.get("active_project")
# project_name = active_project["name"]
project_id = active_project["id"]

# Create project data folder if not created
scenario_folder = Path("data") / "projects" / project_id / "scenarios"
scenario_folder.mkdir(parents=True, exist_ok=True)

# Load all scenarios
all_scenarios = Scenario.load_all(scenario_folder)

# --------------------------
# Session state for selected scenario
# --------------------------
if "selected_scenario" not in st.session_state:
    st.session_state.selected_scenario = None

# --------------------------
# Default view: all scenarios
# --------------------------
if st.session_state.selected_scenario is None:
    all_scenarios_view(all_scenarios, scenario_folder)
else:
    # --------------------------
    # Single scenario dashboard
    # --------------------------
    selected_scenario_id = st.session_state.selected_scenario.id
    s = next(
        (sc for sc in all_scenarios if sc.id == selected_scenario_id),
        None,  # default if not found
    )
    st.session_state.selected_scenario = s

    st.button(
        "⬅ Back to All Scenarios",
        on_click=lambda: st.session_state.update({"selected_scenario": None}),
    )
    st.header(f"Dashboard: {s.name}")

    # --------------------------
    # Top metadata section (cards)
    # --------------------------
    new_metadata_section(project_id, scenario_folder)

    # --------------------------
    # Bottom tabs
    # --------------------------
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "INPUT - Time Allocation",
            "#1. Regionalization Results",
            "#2. Week Distribution Results",
            "#3. Route Optimization Results",
        ]
    )

    with tab1:
        time_allocation_res(s, scenario_folder)
    with tab2:
        regionalization_results(s, scenario_folder)
    with tab3:
        week_distribution_res(s, scenario_folder)
    with tab4:
        fleet_opt_results(s, scenario_folder)


if "editing_scenario_id" not in st.session_state:
    st.session_state.editing_scenario_id = None
if "create_mode" not in st.session_state:
    st.session_state.create_mode = False
if "clone_mode" not in st.session_state:
    st.session_state.clone_mode = False

# --- Show sidebar form ---
editing_scenario = None
if st.session_state.editing_scenario_id:
    editing_scenario = next(
        (s for s in all_scenarios if s.id == st.session_state.editing_scenario_id),
        None,
    )

scenario_sidebar_form(
    project_id,
    scenario_folder,
    is_new=(editing_scenario is None),
    scenario_obj=editing_scenario,
    is_clone=st.session_state.clone_mode,
)
