import json
from datetime import datetime, time
from pathlib import Path
import yaml
from copy import deepcopy
import os

import streamlit as st
import pandas as pd

from session.session import init_session
from utils.auth_utils.guards import require_active_project, require_authentication
from utils.dataframe_utils import to_time

init_session()
require_authentication()
require_active_project()


active_project = st.session_state.get("active_project")
project_name = active_project["name"]
project_id = active_project["id"]

# Create project data folder if not created
project_data_path = Path("data") / "projects" / project_id
project_data_path.mkdir(parents=True, exist_ok=True)
params_path = project_data_path / "input_parameters.json"

stores_path = project_data_path / "stores.json"

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

if os.path.exists(stores_path):
    with open(stores_path, "r") as f:
        stores = pd.DataFrame(json.load(f))
        cities = stores["City"].drop_duplicates().sort_values().to_list()
else:
    cities = []


# Default params
defaults = config["project_default_input_params"].copy()

SHOW_TIME_IN_STORE = False

# Load JSON if present and use as defaults
params = config["project_default_input_params"].copy()
try:
    if params_path.exists():
        loaded = json.loads(params_path.read_text(encoding="utf-8"))
        params = {
            **defaults,
            **{k: v for k, v in loaded.items() if k in defaults},
        }
except Exception:
    pass


st.title("⚙️ Default Input Parameters")
st.markdown(f"for *{project_name}*")

save_button, _, right = st.columns([10, 1, 10])

with save_button:
    # Week schedule (single select box)
    week_schedule_options = ["Sunday - Thursday", "Monday - Friday"]
    week_schedule = st.radio(
        "Week schedule radio",
        options=week_schedule_options,
        index=week_schedule_options.index(params["week_schedule"]),
        help="Select the working week schedule – it can affect the results if traffic is considered",
    )

    # Work day start and end (time selector)
    work_start = st.time_input(
        "Work day start",
        value=to_time(params["work_start"], time(8, 30)),
        help="Start of the working day for all employees",
    )
    work_end = st.time_input(
        "Work day end",
        value=to_time(params["work_end"], time(17, 30)),
        help="End of the working day for all employees",
    )

    # Resulting working days a month – capacity (confirm / override)
    working_days = st.number_input(
        "Working days in a month",
        min_value=0,
        value=params["working_days"],
        step=1,
        help="Model optimization is based on monthly capacity - assuming 4 weeks in a month (e.g., 5 days/week * 4 weeks = 20 days/month). ",
        disabled=True,
    )

    daily_hours = (
        datetime.combine(datetime.today(), work_end)
        - datetime.combine(datetime.today(), work_start)
    ).total_seconds() / 3600

    working_hours_a_month = int(round(working_days * daily_hours, 0))

    # Resulting working hours a month – capacity (confirm / override)
    working_hours = st.number_input(
        "Working hours in a month",
        min_value=0,
        value=working_hours_a_month,
        step=1,
        help="Maximum monthly working hours available, which is calculated based on working days in a month and daily working hours (e.g., 20 days/month * 9 hours/day = 180 hours/month). ",
        disabled=True,
    )

    st.subheader("Operational Delays & Constraints")
    # Lunch break duration (minutes)
    lunch_break = st.number_input(
        "Lunch break duration (minutes)",
        min_value=0,
        max_value=180,
        value=params["lunch_break"],
        step=5,
        help="Duration of lunch break in minutes, e.g. 30 minutes",
    )

    # Average parking time per visit (minutes)
    parking_delay = st.number_input(
        "Average parking time per visit (minutes) - applicable to all the stores",
        min_value=0,
        value=params["parking_delay"],
        step=1,
        help="Estimated parking time delay in minutes per each store visit, e.g. 5 minutes",
    )

    # Initialize session state if not already
    if "other_idle_times" not in st.session_state:
        st.session_state.other_idle_times = params["other_idle_times"]

    # Render inputs for each idle time
    for i, idle in enumerate(st.session_state.other_idle_times):
        cols = st.columns([1, 1, 1])  # name wider, duration narrower
        with cols[0]:
            st.session_state.other_idle_times[i]["name"] = st.text_input(
                f"Other idle time name",
                value=idle.get("name", ""),
                key=f"other_idle_name_{i}",
                help="Provide a descriptive name for this idle time (e.g., Waiting for access, System delays)",
            )
        with cols[1]:
            st.session_state.other_idle_times[i]["applicable_cities"] = st.multiselect(
                "Applicable cities (optional)",
                options=cities,
                default=idle.get("applicable_cities", []),
                key=f"applicable_cities_{i}",
            )
        with cols[2]:
            st.session_state.other_idle_times[i]["minutes"] = st.number_input(
                f"Other idle time (minutes)",
                min_value=0,
                value=idle.get("minutes", 0),
                step=5,
                key=f"other_idle_minutes_{i}",
                help="Duration in minutes for this idle time",
            )

    # Buttons to add/remove idle time rows
    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("➕ Add another idle time"):
            st.session_state.other_idle_times.append({"name": "", "minutes": 0})
            st.rerun()
    with cols[1]:
        if len(st.session_state.other_idle_times) > 0:
            if st.button("➖ Remove last idle time"):
                st.session_state.other_idle_times.pop()
                st.rerun()

    st.subheader("Operational Parameters")
    if SHOW_TIME_IN_STORE:
        time_in_store = st.slider(
            "Estimated time in store – up-time (confirm / override) (%)",
            min_value=0,
            max_value=100,
            value=params["time_in_store"],
            help="Estimated percentage of time spent in stores during working hours. "
            "Confirm or override the calculated value.",
        )
    else:
        time_in_store = params["time_in_store"]

    # Consider delivery dates (boolean checkbox)
    consider_delivery = st.checkbox(
        "Account for delivery dates",
        value=params["consider_delivery"],
        help="Consider delivery dates as hard constraints in the optimization. "
        "If checked, model will schedule visits only on delivery dates.",
    )

    # Road traffic included (boolean checkbox)
    road_traffic = st.checkbox(
        "Account for road traffic",
        value=params["road_traffic"],
        help="Consider road traffic in travel time estimations – can increase API costs",
    )

with right:
    st.markdown(
        """
    **Notes & Assumptions:**
    - These parameters will be used as defaults for all new scenarios within this project
    - Changes here do not affect automatically existing scenarios. You need to rerun specific engines in existing scenarios to apply the changes
    - Model is always optimizing the workload assuming 20 working days in a month (4 weeks)
    - Working hours in a month is calculated based on working days and daily working hours
    - To apply changes in input parameters you need to click "Save parameters" button
    """
    )

input_parameter_dict = {
    "project_id": project_id,
    "working_hours": working_hours,
    "time_in_store": time_in_store,
    "working_days": working_days,
    "work_start": work_start.strftime("%H:%M"),
    "work_end": work_end.strftime("%H:%M"),
    "lunch_break": lunch_break,
    "road_traffic": road_traffic,
    "parking_delay": parking_delay,
    "consider_delivery": consider_delivery,
    "week_schedule": week_schedule,
    "other_idle_times": st.session_state.other_idle_times,
    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
}

save_button, _ = st.columns([1, 3])
with save_button:
    if st.button("Save parameters"):
        # Store all inputs in session state for further use
        st.session_state["input_parameters"] = input_parameter_dict

        # Save input parameters to json
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(input_parameter_dict, f, ensure_ascii=False, indent=2)
        st.rerun()
