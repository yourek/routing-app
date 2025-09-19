import streamlit as st
import time

from utils.auth_utils.auth import authenticate


def require_active_project():
    if (
        "active_project" not in st.session_state
        or not st.session_state["active_project"]
    ):
        st.warning("Please activate a project first from the Projects page.")
        if st.button("Go to Projects"):
            st.switch_page("pages/2_projects.py")
        st.stop()


def require_authentication():
    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        st.subheader("🔐 Login Page")
        with st.form(key="login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

        if submit:
            if authenticate(username, password):
                st.session_state["authenticated"] = True
                st.success("Login successful!")
                time.sleep(1)
                st.rerun()
        st.stop()
