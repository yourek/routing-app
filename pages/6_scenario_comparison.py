import streamlit as st

from session.session import init_session
from utils.auth_utils.guards import require_active_project, require_authentication

init_session()
require_authentication()
require_active_project()

st.title("🔀 Scenario Comparison")
st.write("Work in progress...")
