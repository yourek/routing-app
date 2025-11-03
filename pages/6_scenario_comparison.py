from pathlib import Path
import streamlit as st
from views.scenarios_.scenario_comparison import scenario_comparison

from db.scenarios import Scenario

from session.session import init_session
from utils.auth_utils.guards import require_active_project, require_authentication

init_session()
require_authentication()
require_active_project()

active_project = st.session_state.get("active_project")
project_id = active_project["id"]
scenario_folder = Path("data") / "projects" / project_id / "scenarios"
scenario_folder.mkdir(parents=True, exist_ok=True)

# Load all scenarios
all_scenarios = Scenario.load_all(scenario_folder)

scenario_comparison(all_scenarios, scenario_folder)
