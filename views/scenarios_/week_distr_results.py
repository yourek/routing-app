import pandas as pd
import streamlit as st
import json
import plotly.express as px
import altair as alt

from utils.data_loaders import load_default_input_parameters
from utils.models_common import get_max_month_store_hours


def load_roles_week_distribution_results(scenario, scenario_folder):
    with open(
        scenario_folder / scenario.id / "scenario_metadata.json", "r", encoding="utf-8"
    ) as f:
        scenario_metadata = json.load(f)

    role_names = [role["name"] for role in scenario_metadata["roles"]]
    role_names_sorted = sorted(role_names, key=str.lower)

    roles_data = {}
    for role_name in role_names_sorted:
        try:
            with open(
                scenario_folder
                / scenario.id
                / f"week_dist_res/{role_name} - stores_spread.json",
                "r",
                encoding="utf-8",
            ) as f:
                week_dist = pd.DataFrame(json.load(f))
        except FileNotFoundError:
            st.warning(
                f"Role '{role_name}' has no week distribution results. In order to see results, please re-run the week distribution process."
            )
            continue

        roles_data[role_name] = week_dist

    return roles_data


def week_distribution_res(scenario, scenario_folder):
    if not scenario.regionalization_completed:
        st.info(
            "No week distribution results. Please run the process to see the results."
        )
        return

    roles_data = load_roles_week_distribution_results(scenario, scenario_folder)

    project_params = load_default_input_parameters(scenario.project_id)

    # Single role selector
    role_names_sorted = list(roles_data.keys())
    selected_role = st.selectbox(
        "Select role", role_names_sorted, index=0, help="Choose a role", key="week_dist"
    )
    if not selected_role:
        return

    if role_names_sorted == []:
        st.warning(
            "No roles with week distribution results found. Please run the process to see the results."
        )
        return

    week_dist_res = roles_data[selected_role]
    week_dist_res_agg = (
        week_dist_res.groupby(["Week", "Sales Rep name"])["Visit Duration"]
        .sum()
        .reset_index()
    )

    fig = px.box(
        week_dist_res_agg,
        x="Week",
        y="Visit Duration",
        points="outliers",
        title="Weekly distribution of Time in Store per selected Sales Rep",
    )

    st.plotly_chart(fig)

    week_dist_res_agg_2 = (
        week_dist_res.groupby(["Week", "Sales Rep name"])["Visit Duration"]
        .sum()
        .reset_index()
    )

    # --- Dropdown for week selection ---
    weeks_sorted = sorted(week_dist_res_agg_2["Week"].unique())

    selected_week = st.selectbox("Select Week:", options=weeks_sorted, index=0)

    # Filter data for selected week
    week_df = week_dist_res_agg_2[
        week_dist_res_agg_2["Week"] == selected_week
    ].reset_index(drop=True)

    # --- Horizontal bar chart ---
    chart = (
        alt.Chart(week_df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "Sales Rep name",
                sort="-x",
                axis=alt.Axis(labelAlign="right", labelPadding=10, labelLimit=500),
            ),
            x="Visit Duration",
        )
        .properties(height=max(240, 18 * len(week_df)))
    )

    text = (
        alt.Chart(week_df)
        .mark_text(
            align="left",
            baseline="middle",
            dx=5,
        )
        .encode(
            y=alt.Y("Sales Rep name:N", sort="-x"),
            x=alt.X(f"{"Visit Duration"}:Q"),
            text=alt.Text(f"{"Visit Duration"}:Q", format=".0f"),
        )
    )

    max_month_store_hours = get_max_month_store_hours(project_params, scenario)
    max_week_store_hours = max_month_store_hours / 4
    constant_line = (
        alt.Chart(week_df)
        .mark_rule(strokeDash=[5, 5], color="red")
        .encode(x=alt.datum(max_week_store_hours))
        .properties()
    )

    chart = (chart + constant_line + text).interactive()
    chart_placeholder = st.empty()
    chart_placeholder.altair_chart(
        chart,
        use_container_width=True,
    )
