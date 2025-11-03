import streamlit as st
import extra_streamlit_components as stx
from typing import Optional, Dict

# Setup cookie manager
cookie_manager = stx.CookieManager()


# -------------------------------
# Cookie <-> Session bridge
# -------------------------------
def init_session():
    """Initialize session state from cookies if available, otherwise set defaults."""
    # Active project
    active_project = cookie_manager.get("active_project")
    if "active_project" not in st.session_state:
        st.session_state.active_project = active_project

    active_scenario = cookie_manager.get("active_scenario")
    if "active_scenario" not in st.session_state:
        st.session_state.active_scenario = active_scenario

    # Auth user
    auth_user = cookie_manager.get("auth_user")
    authenticated = cookie_manager.get("authenticated")

    if "auth_user" not in st.session_state:
        st.session_state.auth_user = auth_user

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = authenticated

    # if "ui_flags" not in st.session_state:
    #     st.session_state.ui_flags = {}

    # # Projects (stored as JSON string in cookies if needed)
    # # For now, keep them only in session_state unless you want persistence across tabs
    # if "projects" not in st.session_state:
    #     st.session_state.projects: List[Dict] = []


def set_active_project(project_id: str):
    """Set the active project both in session_state and cookies."""
    st.session_state.active_project = project_id
    if "selected_scenario" in st.session_state:
        st.session_state.selected_scenario = None  # Reset scenario when project changes
    if "other_idle_times" in st.session_state:
        del st.session_state["other_idle_times"]

    cookie_manager.set("active_project", project_id, key="active_project_cookie")


def get_active_project() -> Optional[Dict]:
    """Return the active project object if exists, else None."""
    project_id = st.session_state.get("active_project")
    if not project_id:
        return None
    return project_id


def set_active_scenario(scenario_id: str):
    """Set the active scenario in session_state."""
    st.session_state.active_scenario = scenario_id
    cookie_manager.set("active_scenario", scenario_id, key="active_scenario_cookie")


def get_active_scenario() -> Optional[Dict]:
    """Return the active scenario object if exists, else None."""
    scenario_id = st.session_state.get("active_scenario")
    if not scenario_id:
        return None
    return scenario_id


def get_active_user():
    """Return the active user object if exists, else None."""
    user = st.session_state.get("auth_user", None)
    return user


def set_active_user(user: Dict):
    """Set the active user in session_state."""
    if user:
        st.session_state["auth_user"] = user
        st.session_state["authenticated"] = True
        cookie_manager.set("auth_user", user, key="auth_user_cookie")
        cookie_manager.set("authenticated", True, key="authenticated_cookie")
    else:
        st.session_state["auth_user"] = None
        st.session_state["authenticated"] = False
        cookie_manager.delete("auth_user", key="auth_user_cookie")
        cookie_manager.delete("authenticated", key="authenticated_cookie")
        st.rerun()
