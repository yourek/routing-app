import altair as alt
from copy import deepcopy
import json
import pandas as pd
import streamlit as st
from datetime import datetime

from views.scenarios_.fleet_opt_results import (
    load_roles_fleet_opt_results,
    get_vis_prerequisites,
)


def get_roles_proportions(roles_data):
    roles_proportions = dict()
    for key, value in roles_data.items():
        sales_reps_nb = value["routes_simple"]["Sales Rep name"].nunique()
        visits_nb = value["routes_simple"]["Sales Rep name"].shape[0]
        roles_proportions[key] = {
            "sales_reps_nb": sales_reps_nb,
            "visits_nb": visits_nb,
        }
    return roles_proportions


def scenario_comparison(all_scenarios, scenario_folder):
    project_params_path = scenario_folder.parent / "input_parameters.json"
    with open(project_params_path, "r", encoding="utf-8") as f:
        project_params = json.load(f)

    created_scenarios_nb = len(all_scenarios)
    scenarios_w_rgn_res = [s for s in all_scenarios if s.regionalization_completed]
    scenarios_w_week_dist_res = [
        s for s in all_scenarios if s.week_distribution_completed
    ]
    scenarios_w_fleet_opt_res = [
        s for s in all_scenarios if s.route_optimization_completed
    ]

    scenarios_counts = [
        created_scenarios_nb,
        len(scenarios_w_rgn_res),
        len(scenarios_w_week_dist_res),
        len(scenarios_w_fleet_opt_res),
    ]
    card_addins = [
        "scenarios created",
        "scenarios with completed Regionalization",
        "scenarios with completed Week Distribution",
        "scenarios with completed Route Optimization",
    ]

    cols = st.columns(4)
    for i, col in enumerate(cols):
        with col:
            st.markdown(
                f"""
                <div style="
                    background-color:#f8f9fa;
                    padding:1.5rem;
                    border-radius:0.75rem;
                    box-shadow:0 2px 4px rgba(0,0,0,0.1);
                    text-align:center;
                    height:150px;
                    display:flex;
                    flex-direction:column;
                    justify-content:center;
                ">
                    <h4 style="margin-bottom:0.5rem; font-size:2rem;">{scenarios_counts[i]}</h4>
                    <p style="margin:0; color:#555;">{card_addins[i]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # === Build FTE data for each scenario ===
    fte_chart_input = dict()
    for s in scenarios_w_rgn_res:
        fte_chart_input_curr = dict()
        scenario_folder_curr = scenario_folder / s.id
        scenario_metadata_curr_path = scenario_folder_curr / "scenario_metadata.json"

        with open(scenario_metadata_curr_path, "r") as f:
            scenario_metadata_curr = json.load(f)

        for role in scenario_metadata_curr.get("roles", []):
            try:
                rgn_res_path = scenario_folder_curr / "rgn_res" / f"{role['name']}.json"
                with open(rgn_res_path, "r", encoding="utf-8") as f:
                    rgn_res_curr = pd.DataFrame(json.load(f))
                fte_chart_input_curr[role["name"]] = rgn_res_curr[
                    "Sales Rep name"
                ].nunique()
            except FileNotFoundError:
                pass
        fte_chart_input[s.name] = deepcopy(fte_chart_input_curr)

    # === Prepare chart dataframe ===
    data = []
    for scenario, roles in fte_chart_input.items():
        for role, fte in roles.items():
            data.append({"Scenario": scenario, "Role": role, "FTEs": fte})

    df = pd.DataFrame(data)
    if df.empty:
        st.warning(
            "No regionalization results. Please run the process to see the results."
        )
        return

    df = df.groupby(["Scenario", "Role"], as_index=False)["FTEs"].sum()
    df["ScenarioTotal"] = df.groupby("Scenario")["FTEs"].transform("sum")
    df["Percent"] = df["FTEs"] / df["ScenarioTotal"] * 100

    # === Control stacking order explicitly ===
    # Sort roles alphabetically or define custom order (consistent across chart + labels)
    role_order = sorted(df["Role"].unique().tolist())

    # Apply consistent sort to DF for label position calc
    df = df.sort_values(
        ["Scenario", "Role"],
        key=lambda x: x.map({r: i for i, r in enumerate(role_order)}),
    )

    # Compute manual stacking for label positioning
    df["y0"] = df.groupby("Scenario")["FTEs"].cumsum() - df["FTEs"]
    df["y_mid"] = df["y0"] + df["FTEs"] / 2
    df["LabelText"] = df.apply(
        lambda r: f"{r['FTEs']:.0f} ({r['Percent']:.0f}%)", axis=1
    )

    # === Build Altair chart ===
    st.markdown("<br><br>", unsafe_allow_html=True)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Scenario:N", title="Scenario", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("sum(FTEs):Q", title="Total FTEs", stack="zero"),
            color=alt.Color("Role:N", title="Role", sort=role_order),
            order=alt.Order("Role:N", sort="ascending"),
            tooltip=[
                "Scenario",
                "Role",
                "FTEs",
                alt.Tooltip("Percent:Q", format=".1f"),
            ],
        )
        .properties(title="FTE Distribution by Scenario", width=600, height=400)
    )

    # Correct label alignment using same stack order
    labels = (
        alt.Chart(df)
        .mark_text(fontSize=12, color="black", baseline="middle", align="center")
        .encode(
            x=alt.X("Scenario:N"),
            y=alt.Y("y_mid:Q"),
            text="LabelText:N",
        )
    )

    totals_df = (
        df[["Scenario", "ScenarioTotal"]]
        .drop_duplicates()
        .assign(y_pos=lambda d: d["ScenarioTotal"] + 3)  # small offset above bar
    )

    total_labels = (
        alt.Chart(totals_df)
        .mark_text(
            fontSize=13,
            fontWeight="bold",
            color="black",
            baseline="bottom",
            dy=-2,  # lift text slightly above bar
        )
        .encode(
            x=alt.X("Scenario:N"),
            y=alt.Y("y_pos:Q"),
            text=alt.Text("ScenarioTotal:Q", format=".0f"),
        )
    )

    st.altair_chart(chart + labels + total_labels, use_container_width=True)

    if not scenarios_w_fleet_opt_res:
        return

    # REFACTOR
    time_alloc_chart_input = dict()
    for s in scenarios_w_fleet_opt_res:
        roles_data = load_roles_fleet_opt_results(s, scenario_folder)
        time_alloc_chart_input_curr = dict()
        for role in roles_data.keys():
            time_alloc_chart_input_curr[role] = (
                roles_data[role]["routes_simple"]["Time in Store (min)"].sum() / 60
            )
        time_alloc_chart_input[s.name] = deepcopy(time_alloc_chart_input_curr)

    # start
    data = []
    for scenario, roles in time_alloc_chart_input.items():
        for role, time_in_store in roles.items():
            data.append(
                {"Scenario": scenario, "Role": role, "Time in store (h)": time_in_store}
            )

    df = pd.DataFrame(data)
    if df.empty:
        st.warning(
            "No route optimization results. Please run the process to see the results."
        )
        return

    df = df.groupby(["Scenario", "Role"], as_index=False)["Time in store (h)"].sum()
    df["ScenarioTotal"] = df.groupby("Scenario")["Time in store (h)"].transform("sum")
    df["Percent"] = df["Time in store (h)"] / df["ScenarioTotal"] * 100

    # === Control stacking order explicitly ===
    # Sort roles alphabetically or define custom order (consistent across chart + labels)
    role_order = sorted(df["Role"].unique().tolist())

    # Apply consistent sort to DF for label position calc
    df = df.sort_values(
        ["Scenario", "Role"],
        key=lambda x: x.map({r: i for i, r in enumerate(role_order)}),
    )

    # Compute manual stacking for label positioning
    df["y0"] = (
        df.groupby("Scenario")["Time in store (h)"].cumsum() - df["Time in store (h)"]
    )
    df["y_mid"] = df["y0"] + df["Time in store (h)"] / 2
    df["LabelText"] = df.apply(
        lambda r: f"{r['Time in store (h)']:.0f} ({r['Percent']:.0f}%)", axis=1
    )

    # === Build Altair chart ===
    st.markdown("<br><br>", unsafe_allow_html=True)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Scenario:N", title="Scenario", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "sum(Time in store (h)):Q",
                title="Total time in store (h)",
                stack="zero",
            ),
            color=alt.Color("Role:N", title="Role", sort=role_order),
            order=alt.Order("Role:N", sort="ascending"),
            tooltip=[
                "Scenario",
                "Role",
                "Time in store (h)",
                alt.Tooltip("Percent:Q", format=".1f"),
            ],
        )
        .properties(title="Time Allocated by Scenario", width=600, height=400)
    )

    # Correct label alignment using same stack order
    labels = (
        alt.Chart(df)
        .mark_text(fontSize=12, color="black", baseline="middle", align="center")
        .encode(
            x=alt.X("Scenario:N"),
            y=alt.Y("y_mid:Q"),
            text="LabelText:N",
        )
    )

    totals_df = (
        df[["Scenario", "ScenarioTotal"]]
        .drop_duplicates()
        .assign(y_pos=lambda d: d["ScenarioTotal"] + 3)  # small offset above bar
    )

    total_labels = (
        alt.Chart(totals_df)
        .mark_text(
            fontSize=13,
            fontWeight="bold",
            color="black",
            baseline="bottom",
            dy=-2,  # lift text slightly above bar
        )
        .encode(
            x=alt.X("Scenario:N"),
            y=alt.Y("y_pos:Q"),
            text=alt.Text("ScenarioTotal:Q", format=".0f"),
        )
    )

    st.altair_chart(chart + labels + total_labels, use_container_width=True)
    # stop

    workday_start = datetime.strptime(project_params["work_start"], "%H:%M").time()
    workday_end = datetime.strptime(project_params["work_end"], "%H:%M").time()
    # TODO: delete "/ 4"
    total_fte_capacity = (
        (
            datetime.combine(datetime.today(), workday_end)
            - datetime.combine(datetime.today(), workday_start)
        ).total_seconds()
        / 60
        * 20
        / 4
    )

    metric_names = [
        "Utilization (in %)",
        "Travel Distance (per visit, in km)",
        "Travel Time (per visit, in minutes)",
        "Avg. visit time (in minutes)",
        "Daily Visits (per sales rep, # of visits)",
        "Unallocated time (per visit, in minutes)",
    ]
    for i, metric in enumerate(metric_names):
        metrics = dict()
        metrics[metric] = {}
        for s in scenarios_w_fleet_opt_res:
            roles_data = load_roles_fleet_opt_results(s, scenario_folder)
            (
                _,
                kpis_table_input,
                _,
                _,
                _,
                _,
            ) = get_vis_prerequisites(roles_data, total_fte_capacity)
            roles_proportions = get_roles_proportions(roles_data)
            sales_reps_nb_total = sum(
                [value["sales_reps_nb"] for value in roles_proportions.values()]
            )
            visits_nb_total = sum(
                [value["visits_nb"] for value in roles_proportions.values()]
            )

            metric_value = 0.0
            for key, value in kpis_table_input.items():
                if metric == "Utilization (in %)":
                    metric_value += (
                        value[metric]
                        * roles_proportions[key]["sales_reps_nb"]
                        / sales_reps_nb_total
                    )
                else:
                    metric_value += (
                        value[metric]
                        * roles_proportions[key]["visits_nb"]
                        / visits_nb_total
                    )
            metrics[metric][s.name] = metric_value

        df = pd.DataFrame(
            [{"Scenario": k, metric: v} for k, v in metrics[metric].items()]
        )

        if "Avg. visit time (in minutes)" in df.columns:
            df = df.rename(
                columns={"Avg. visit time (in minutes)": "Avg visit time (in minutes)"}
            )
            metric = "Avg visit time (in minutes)"

        # Altair chart
        chart_metric = (
            alt.Chart(df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Scenario:N", title="Scenario", axis=alt.Axis(labelAngle=0)),
                y=alt.Y(f"{metric}:Q", title=metric),
                color=alt.Color("Scenario:N", legend=None),
                tooltip=["Scenario", f"{metric}:Q"],
            )
            .properties(title=metric, width=400)
            .configure_title(
                fontSize=16,
                anchor="start",  # 'middle' is default
                offset=20,  # pushes the title down so it’s not cut
            )
        )

        # Display in Streamlit
        if not (i % 2):
            cols = st.columns(2)
            col = cols[0]
        else:
            col = cols[1]
        with col:
            st.altair_chart(chart_metric, use_container_width=False)
