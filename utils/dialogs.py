import time
import streamlit as st
from utils.regionalization import estimate_n_clusters_all, regionalization


# @st.dialog("Confirmation")
# def confirmation_dialog():
#     st.write("Are you sure you want to delete this scenario?")
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Yes"):
#             st.session_state.confirm_delete_scenario = None
#             st.success("Scenario deleted successfully.")
#     with col2:
#         if st.button("No"):
#             st.session_state.confirm_delete_scenario = None
#             st.info("Deletion cancelled.")


# @st.dialog("Engine is running test changes?", dismissible=True)
# def engine_running_dialog(engine: str):
#     st.write(f"{engine} is currently running. Please wait until it finishes.")
#     with st.spinner("Engine is running..."):
#         st.write(f"{engine} is currently running. Please wait until it finishes.")
#         time.sleep(5)
#     if st.button("OK"):
#         st.session_state.engine_running = False
#         st.info("You can now proceed with other actions.")


@st.dialog("Engine is running", dismissible=False)
def engine_running_dialog(
    engine: str, task_fn=None, session_state_key="engine_running"
):
    st.write(f"⚙️ {engine} is currently running. Please wait until it finishes.")

    if session_state_key not in st.session_state:
        st.session_state[session_state_key] = {
            "running": True,
            "done": False,
            "result": None,
        }

    state = st.session_state[session_state_key]

    if state["running"] and not state["done"]:
        if task_fn is not None:
            with st.spinner(
                "Engine is running... It may take a few minutes. Do not refresh the page."
            ):
                state["result"] = task_fn()
            state["done"] = True
            state["running"] = False
        else:
            state["result"] = "No task provided."
            state["done"] = True
            state["running"] = False

    # Show result and OK button
    if state["done"]:
        st.success(f"{engine} has finished running.")
        if st.button("OK"):
            del st.session_state[session_state_key]  # cleanup dialog state
            st.rerun()


@st.dialog("Regionalization engine", dismissible=False)
def regionalization_dialog(scenario_folder, scenario):
    state = st.session_state.setdefault(
        "regionalization_engine_running", {"phase": "input", "done": False}
    )

    if state["phase"] == "input":
        # run FTE estimator
        estimated_fte_dict = estimate_n_clusters_all(scenario_folder, scenario)

        # Store in session state so it's available after rerun
        st.session_state["estimated_fte_dict"] = estimated_fte_dict

        st.info(
            "Estimated minimal number of FTEs below. You can adjust them if needed."
        )
        for role, value in estimated_fte_dict.items():
            new_value = st.number_input(
                f"FTEs for {role}",
                min_value=value,
                step=1,
                value=value,
                key=f"fte_{role}",
            )
            estimated_fte_dict[role] = new_value

        if st.button("Run the engine"):
            # Update state
            st.session_state["estimated_fte_dict"] = estimated_fte_dict
            state["phase"] = "processing"
            st.rerun()

    elif state["phase"] == "processing":
        st.info("Running regionalization engine...")

        # Retrieve from session state
        estimated_fte_dict = st.session_state.get("estimated_fte_dict", {})

        with st.spinner("Regionalization in progress..."):
            regionalization(scenario_folder, scenario, estimated_fte_dict)

        state["phase"] = "done"
        st.rerun()

    elif state["phase"] == "done":
        st.success("Regionalization completed successfully!")
        st.write("You can now review the results in the dashboard.")

        if st.button("OK"):
            st.session_state.pop("regionalization_engine_running", None)
            st.session_state.pop("estimated_fte_dict", None)
            st.session_state.pop("show_regionalization_dialog", None)
            st.info("Dialog closed.")
            st.rerun()
