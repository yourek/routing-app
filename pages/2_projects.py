import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from db.projects import delete_project, load_projects, save_project
from session.session import (
    get_active_project,
    get_active_user,
    init_session,
    set_active_project,
)
import streamlit as st
from utils.auth_utils.guards import require_authentication


init_session()
require_authentication()

st.title("🚀 Projects")
# TODO change it in the future to real user.
auth_user = get_active_user()

USER = auth_user["username"] if auth_user else "unknown user"


def new_project_draft(author: str | None = None):
    """Return an unsaved project dict. Nothing is written to disk yet."""
    pid = f"project_{uuid4().hex[:8]}"
    draft = {
        "id": pid,
        "name": "New Project",
        "description": "Add a short description…",
        "author": author or "anonymous",
        # created_at will be set on save
    }
    return draft


def activate_project(pid: str):
    # Ensure only one active at a time
    projects = load_projects()
    selected_name = ""
    for p in projects:
        if p["id"] == pid:
            selected_name = p["name"]
            selected_project = p
        save_project(p)

    # Store the active project in session state
    st.session_state["project_name"] = selected_name
    st.session_state["project"] = selected_project

    # Remove stores table in project is changed
    if "stores_table" in st.session_state:
        del st.session_state["stores_table"]


# # Top controls
cols = st.columns([1, 1, 1])
with cols[0]:
    if st.button(
        "\\+ Create a new project", key="create_btn", use_container_width=False
    ):
        # Prepare an unsaved draft and open the edit dialog
        draft = new_project_draft(author=USER)
        st.session_state["edit_id"] = draft["id"]
        # holds values while editing (unsaved)
        st.session_state["edit_data"] = draft
        st.session_state["is_new"] = True
        # no rerun: the edit sidebar below will render this run


# Projects grid
projects = load_projects()

if not projects:
    st.info("No projects yet. Click **Create a new project** to get started.")
else:
    # 3 cards per row on wide screens
    n_per_row = 3
    rows = [projects[i : i + n_per_row] for i in range(0, len(projects), n_per_row)]
    for row in rows:
        cols = st.columns(n_per_row, gap="large")
        for col, proj in zip(cols, row):
            with col:
                with st.container(border=True):
                    st.markdown(
                        '<div class="project-card">',
                        unsafe_allow_html=True,
                    )

                    # Top: name and active badge
                    name_line = f"### {proj.get('name', '<name>')}"

                    st.markdown(name_line, unsafe_allow_html=True)

                    # Middle: fields
                    st.markdown(
                        f"""
                        <div class="project-fields">
                          <div>{proj.get('description', '<description>')}</div>
                          <div style="opacity:.7;margin-top:8px;">created: {proj.get('created_at', '<create_date>')}</div>
                          <div style="opacity:.7;">author: {proj.get('author', '<author>')}</div>
                          <div style="opacity:.7;">project id: {proj.get('id', '<id>').split("_")[-1]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Bottom: actions
                    st.markdown('<div class="project-actions">', unsafe_allow_html=True)
                    a1, a2, a3 = st.columns(3)

                    with a1:
                        if st.button(
                            "Activate Project",
                            key=f"act_{proj['id']}",
                            use_container_width=True,
                        ):
                            # activate_project(proj["id"])
                            # set_active_project(proj["id"])
                            set_active_project(proj)
                            st.switch_page("pages/3_add_manage_stores.py")
                            st.rerun()

                    with a2:
                        if st.button(
                            "Edit", key=f"edit_{proj['id']}", use_container_width=True
                        ):
                            st.session_state["edit_id"] = proj["id"]
                            st.session_state["edit_data"] = proj.copy()
                            st.session_state["is_new"] = False

                    with a3:
                        if st.button(
                            "Delete", key=f"del_{proj['id']}", use_container_width=True
                        ):
                            st.session_state["delete_confirm_id"] = proj["id"]
                    # close actions
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)  # close card


# Edit / Create dialog
edit_id = st.session_state.get("edit_id")
if edit_id:
    # Determine whether this is a new (unsaved) project or an existing one
    is_new = st.session_state.get("is_new", False)

    # For existing: find from disk; for new: use session draft
    proj = None
    if is_new:
        proj = st.session_state.get("edit_data", {})
    else:
        proj = next((p for p in projects if p["id"] == edit_id), None)
        if proj is None:
            # Fallback: use any cached edit data if present
            proj = st.session_state.get("edit_data", {})

    if proj:
        with st.sidebar:
            st.markdown("### " + ("Create Project" if is_new else "Edit Project"))
            with st.form(f"form_edit_{edit_id}", clear_on_submit=False):
                name = st.text_input("Name", value=proj.get("name", ""))
                desc = st.text_area(
                    "Description", value=proj.get("description", ""), height=120
                )
                author = st.text_input("Author", value=proj.get("author", ""))

                primary_label = "Create project" if is_new else "Save changes"
                submitted = st.form_submit_button(
                    primary_label, use_container_width=True, type="primary"
                )
                cancel = st.form_submit_button("Cancel", use_container_width=True)

                if submitted:
                    # Basic validation
                    if not name.strip():
                        st.warning("Please provide a project name.")
                        st.stop()

                    # Build the final dict for saving
                    payload = {
                        "id": edit_id,
                        "name": name.strip(),
                        "description": desc.strip(),
                        "author": author.strip() or "anonymous",
                        "created_at": proj.get("created_at")
                        or datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }

                    # Save first (must exist on disk before activation toggles others)
                    save_project(payload)

                    # If set active, ensure others are inactive

                    # Cleanup session
                    st.session_state["edit_id"] = None
                    st.session_state["edit_data"] = None
                    st.session_state["is_new"] = False

                    st.success("Project saved.")

                    set_active_project(payload)
                    st.switch_page("pages/3_add_manage_stores.py")
                    st.rerun()

                elif cancel:
                    # Discard draft / close editor
                    st.session_state["edit_id"] = None
                    st.session_state["edit_data"] = None
                    st.session_state["is_new"] = False
                    st.rerun()


# Delete confirm
del_id = st.session_state.get("delete_confirm_id")
if del_id:
    proj = next((p for p in projects if p["id"] == del_id), None)
    if proj:
        with st.sidebar:
            st.markdown("### Delete Project")
            st.warning(f"Delete **{proj['name']}**? This action cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button(
                    "Yes, delete", key=f"yes_del_{del_id}", use_container_width=True
                ):
                    delete_project(del_id)
                    st.session_state["delete_confirm_id"] = None
                    st.rerun()
            with c2:
                if st.button(
                    "Cancel", key=f"no_del_{del_id}", use_container_width=True
                ):
                    st.session_state["delete_confirm_id"] = None
                    st.rerun()
