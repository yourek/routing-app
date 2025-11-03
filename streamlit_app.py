import streamlit as st

from session.session import init_session
from utils.auth_utils.auth import logout
from utils.auth_utils.guards import require_authentication
from utils.styles import load_styles

import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "config/application_default_credentials.json"
)

load_styles()
init_session()
require_authentication()

st.set_page_config(
    page_title="Ferrero Route Optimization Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.logo("assets/logo.png", size="medium", link=None, icon_image=None)

active_project = st.session_state.get("active_project", None)
project_name = f"Active project: {active_project['name']}" if active_project else ""

# Top navbar block
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <style>
    :root {{
        --custom-navbar-height: 60px;
    }}

    [data-testid="stHeader"] {{
        margin-top: var(--custom-navbar-height);
    }}

    .custom-navbar {{
        position: fixed;
        top: 0; left: 0; right: 0;
        height: var(--custom-navbar-height);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 16px;
        background: #212121;
        border-bottom: 1px solid #e6e6e6;
        z-index: 1001;
        font-size: 16px;
    }}

    .custom-navbar .left {{
        font-weight: 700;
        display: flex; align-items: center; gap: 8px;
    }}
    .custom-navbar .center {{
        flex: 1; text-align: center; font-weight: 600;
    }}
    .custom-navbar .right {{
        display: flex; align-items: center; gap: 14px;
    }}

    .main .block-container {{
        padding-top: calc(var(--custom-navbar-height) + 20px);
    }}
    </style>

    <div class="custom-navbar">
    <div class="left" style="font-size:28px; color:white;"></div>

    <div class="center" style="font-size:28px; color:white;">{project_name}</div>
    <div class="right">

    </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Place a placeholder for the button
button_placeholder = st.empty()

# Inject button inside the placeholder
with button_placeholder.container():
    if st.button("👤 Logout", key="absolute-button-logout"):
        logout()
        st.success("Logged out successfully!")


## PAGES

instructions = st.Page("pages/1_instructions.py", title="Instructions")
projects = st.Page("pages/2_projects.py", title="Projects")
add_manage_stores = st.Page("pages/3_add_manage_stores.py", title="Add & Manage Stores")
default_input_parameters = st.Page(
    "pages/4_default_input_parameters.py", title="Default Input Parameters"
)
add_manage_scenarios = st.Page(
    "pages/5_add_manage_scenarios.py", title="Add & Manage Scenarios"
)
scenario_comparison = st.Page(
    "pages/6_scenario_comparison.py", title="Scenario Comparison"
)
documentation = st.Page("pages/7_documentation.py", title="Documentation")

pg = st.navigation(
    pages=[
        instructions,
        projects,
        add_manage_stores,
        default_input_parameters,
        add_manage_scenarios,
        scenario_comparison,
        documentation,
    ]
)

pg.run()
