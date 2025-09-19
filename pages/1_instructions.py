import streamlit as st

st.title("📖 Instructions")

st.write("")
# Projects
st.markdown(
    """
    #### Step 1: Create a Project
    ##### Objective
    Creates a new project for route optimization, allowing users to manage multiple datasets and scenarios. Each project can be a different region or business case.
    You can have only one dataset which is used for optimization.
    """
)
if st.button("Go to Projects"):
    st.switch_page("pages/2_projects.py")
st.divider()

# Stores
st.markdown(
    """
    #### Step 2: Add & Manage Stores
    ##### Objective
    Batch upload and edit store data on a fly, including locations, names and delivery dates.
    """
)
if st.button("Go to Stores Managers"):
    st.switch_page("pages/3_add_manage_stores.py")

st.divider()

# Default Input Parameters
st.markdown(
    """
    #### Step 3: Upload Default Input Parameters
    ##### Objective
    Setup default parameters for route optimization, such as number of working hours a month, estimated time in store, working day windows, inclusion of delivery dates, etc.
    Each project can have only one set of default input parameters which is shared across all the scenarios
    """
)
if st.button("Go to Default Input Parameters"):
    st.switch_page("pages/4_default_input_parameters.py")
st.divider()

# Add & Manage Scenarios
st.markdown(
    """
    #### Step 4: Add & Manage Scenarios
    ##### Objective
    Create and manage multiple scenarios within a project, each with its own set of roles (e.g. SR + Merchendiser, or junior SR, senior SR, Merchendiser, etc.).
    For each role assigned within scenario, manage number of visits and time spent in store for each store.
    
    """
)
if st.button("Go to Add & Manage Scenarios"):
    st.switch_page("pages/5_add_manage_scenarios.py")
st.divider()

# Scenario Comparison
st.markdown(
    """
    #### Step 5: Scenario Comparison
    ##### Objective
    Compare different scenarios within a project to evaluate their performance and identify the best approach.
    """
)
if st.button("Go to Scenario Comparison"):
    st.switch_page("pages/6_scenario_comparison.py")
