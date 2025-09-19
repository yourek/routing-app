import json
from datetime import datetime, time
from pathlib import Path

import streamlit as st

from session.session import init_session
from utils.auth_utils.guards import require_active_project
from utils.dataframe_utils import to_time

init_session()
require_active_project()

active_project = st.session_state.get("active_project")
project_name = active_project["name"]
project_id = active_project["id"]

# Create project data folder if not created
project_data_path = Path("data") / "projects_data" / project_id
project_data_path.mkdir(parents=True, exist_ok=True)
params_path = project_data_path / "input_parameters.json"

# Default params
defaults = {
    "working_hours": 170,
    "time_in_store": 83,
    "working_days": 20,
    "work_start": "08:30",
    "work_end": "17:30",
    "lunch_break": 30,
    "parking_delay": 5,
    "week_schedule": "Sunday - Thursday",
    "consider_delivery": True,
    "road_traffic": False,
}

# Load JSON if present and use as defaults
params = defaults.copy()
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

save_button, right = st.columns([1, 1])

with save_button:
    # Week schedule (single select box)

    st.subheader("Schedule & Capacity Parameters")
    week_schedule = st.selectbox(
        "Week schedule",
        options=["Sunday - Thursday", "Monday - Friday"],
        index=["Sunday - Thursday", "Monday - Friday"].index(params["week_schedule"]),
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
        "Default working days in a month",
        min_value=0,
        value=params["working_days"],
        step=1,
        help="Base capacity in days per month. Adjust if needed (e.g., default 20, but actual is 22). "
        "Model will scale results accordingly.",
    )

    # Resulting working hours a month – capacity (confirm / override)
    working_hours = st.number_input(
        "Default working hours in a month",
        min_value=0,
        value=params["working_hours"],
        step=1,
        help="Maximum monthly working hours available (e.g., 40 hours/week * 4 weeks = 160 hours/month). "
        "Adjust if needed to override the calculated capacity.",
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
        "Average parking time per visit (minutes)",
        min_value=0,
        value=params["parking_delay"],
        step=1,
        help="Estimated parking time delay in minutes per each store visit, e.g. 5 minutes",
    )

    # Initialize session state if not already
    if "other_idle_times" not in st.session_state:
        st.session_state.other_idle_times = []  # start empty

    # Render inputs for each idle time
    for i, idle in enumerate(st.session_state.other_idle_times):
        cols = st.columns([2, 1])  # name wider, duration narrower
        with cols[0]:
            st.session_state.other_idle_times[i]["name"] = st.text_input(
                f"Other idle time name",
                value=idle.get("name", ""),
                key=f"other_idle_name_{i}",
                help="Provide a descriptive name for this idle time (e.g., Waiting for access, System delays)",
            )
        with cols[1]:
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

    # # Initialize session state if not already
    # if "other_idle_times" not in st.session_state:
    #     st.session_state.other_idle_times = params.get(
    #         "other_idle_times", [0]
    #     )  # start with one

    # # Render inputs for each idle time
    # for i in range(len(st.session_state.other_idle_times)):
    #     st.session_state.other_idle_times[i] = st.number_input(
    #         f"Other idle time {i+1} (minutes)",
    #         min_value=0,
    #         value=st.session_state.other_idle_times[i],
    #         step=5,
    #         key=f"other_idle_time_{i}",
    #         help="Additional idle time not captured elsewhere",
    #     )

    # # Buttons to add/remove idle time rows
    # cols = st.columns([1, 1])
    # with cols[0]:
    #     if st.button("➕ Add another idle time"):
    #         st.session_state.other_idle_times.append(0)
    #         st.rerun()
    # with cols[1]:
    #     if len(st.session_state.other_idle_times) >= 1:
    #         if st.button("➖ Remove last idle time"):
    #             st.session_state.other_idle_times.pop()
    #             st.rerun()

    # Resulting estimated time in store – up-time (confirm / override)

    st.subheader("Operational Parameters")
    time_in_store = st.slider(
        "Estimated time in store – up-time (confirm / override) (%)",
        min_value=0,
        max_value=100,
        value=params["time_in_store"],
        help="Estimated percentage of time spent in stores during working hours. "
        "Confirm or override the calculated value.",
    )

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


# st.subheader("Current Parameters")
# st.json(st.session_state.get("input_parameters", input_parameter_dict), expanded=True)
