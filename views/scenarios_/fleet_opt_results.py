import json
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
from datetime import datetime

from utils.dataframe_utils import download_dataframe, filter_dataframe


def load_roles_fleet_opt_results(scenario, scenario_folder):
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
                / "fleet_opt/final"
                / f"{role_name} - routes_simple.json",
                "r",
                encoding="utf-8",
            ) as f:
                routes_simple_curr = pd.DataFrame(json.load(f))
            with open(
                scenario_folder
                / scenario.id
                / "fleet_opt/final"
                / f"{role_name} - skipped_shipments.json",
                "r",
                encoding="utf-8",
            ) as f:
                skipped_shipments_curr = pd.DataFrame(json.load(f))
        except FileNotFoundError:
            st.warning(
                f"Role '{role_name}' has no fleet optimization results. In order to see results, please re-run the fleet optimization process."
            )
            continue

        roles_data[role_name] = {
            "routes_simple": routes_simple_curr,
            "skipped_shipments": skipped_shipments_curr,
        }

    return roles_data


def get_vis_prerequisites(roles_data, total_fte_capacity):
    time_alloc_chart_input = dict()
    kpis_table_input = dict()
    skip_ship_summary_tab_input = dict()
    skip_ship_vis_input = dict()
    skip_ship_vis_input_prettified = dict()

    skip_ship_vis_prettified_cols = [
        "Customer",
        "Role",
        "Sales Rep name",
        "Week",
        "Distribution Days",
        "Visit Duration (in minutes)",
    ]

    for role in roles_data.keys():
        routes_simple_curr = roles_data[role]["routes_simple"]
        skipped_shipments_curr = roles_data[role]["skipped_shipments"]

        time_in_store = routes_simple_curr["Time in Store (min)"].mean()
        travel_time = (
            routes_simple_curr["Travel to Store (min)"]
            + routes_simple_curr["Park and Walk (min)"]
        ).mean()
        unallocated_time = (
            total_fte_capacity * routes_simple_curr["Sales Rep name"].nunique()
            - routes_simple_curr["Time in Store (min)"].sum()
            - routes_simple_curr["Travel to Store (min)"].sum()
            - routes_simple_curr["Park and Walk (min)"].sum()
        ) / routes_simple_curr.shape[0]

        utilization = routes_simple_curr["Time in Store (min)"].sum() / (
            total_fte_capacity * routes_simple_curr["Sales Rep name"].nunique()
        )
        avg_distance = routes_simple_curr["Travel to Store (km)"].mean()
        travel_time = (
            routes_simple_curr["Travel to Store (min)"]
            + routes_simple_curr["Park and Walk (min)"]
        ).mean()
        avg_visit_time = routes_simple_curr["Time in Store (min)"].mean()
        # TODO: delete "* 4"
        daily_visits = (
            routes_simple_curr.shape[0]
            / routes_simple_curr["Sales Rep name"].nunique()
            / 20
            * 4
        )

        time_alloc_chart_input[role] = {
            "time_in_store": time_in_store,
            "travel_time": travel_time,
            "unallocated_time": unallocated_time,
        }
        kpis_table_input[role] = {
            "Utilization (in %)": utilization,
            "Idle time (Incl. Travel / Parking)": 1 - utilization,
            "Travel Distance (per visit, in km)": avg_distance,
            "Travel Time (per visit, in minutes)": travel_time,
            "Avg. visit time (in minutes)": avg_visit_time,
            "Daily Visits (per sales rep, # of visits)": daily_visits,
            "Unallocated time (per visit, in minutes)": unallocated_time,
        }
        skipped_vis_nb = skipped_shipments_curr.shape[0]
        skipped_vis_perc = skipped_vis_nb / routes_simple_curr.shape[0]
        skipped_vis_perc_str = f"{skipped_vis_perc:.0%}"

        skip_ship_summary_tab_input[role] = {
            "Number of unallocated visits": f"{skipped_vis_nb} ({skipped_vis_perc_str})",
            "Duration of unallocated visits (in hours)": (
                skipped_shipments_curr["Visit Duration"].sum()
                if skipped_shipments_curr.shape[0]
                else 0.0
            ),
        }
        if skipped_shipments_curr.shape[0]:
            skipped_shipments_curr["Sales Rep name"] = skipped_shipments_curr[
                "group"
            ].str.rsplit("_", n=1, expand=True)[0]
            skipped_shipments_curr_prettified = skipped_shipments_curr.copy()
            skipped_shipments_curr_prettified["Role"] = role
            skipped_shipments_curr_prettified["Visit Duration (in minutes)"] = (
                skipped_shipments_curr_prettified["Visit Duration"] * 60
            )
            skipped_shipments_curr_prettified["Week"] = (
                skipped_shipments_curr_prettified["group"].str.rsplit(
                    "_", n=1, expand=True
                )[1]
            )
            skipped_shipments_curr_prettified["Week"] = (
                skipped_shipments_curr_prettified["Week"].str.title()
            )
            skipped_shipments_curr_prettified = skipped_shipments_curr_prettified[
                skip_ship_vis_prettified_cols
            ]
        else:
            skipped_shipments_curr_prettified = skipped_shipments_curr.copy()
        skip_ship_vis_input[role] = skipped_shipments_curr
        skip_ship_vis_input_prettified[role] = skipped_shipments_curr_prettified

    return (
        time_alloc_chart_input,
        kpis_table_input,
        skip_ship_summary_tab_input,
        skip_ship_vis_input,
        skip_ship_vis_input_prettified,
        skip_ship_vis_prettified_cols,
    )


def fleet_opt_results(scenario, scenario_folder):
    if not scenario.route_optimization_completed:
        st.info(
            "No route optimization results. Please run the process to see the results."
        )
        return

    stores_path = scenario_folder.parent / "stores.json"
    project_params_path = scenario_folder.parent / "input_parameters.json"
    with open(project_params_path, "r", encoding="utf-8") as f:
        project_params = json.load(f)
    with open(stores_path, "r", encoding="utf-8") as f:
        stores = pd.DataFrame(json.load(f))

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

    roles_data = load_roles_fleet_opt_results(scenario, scenario_folder)

    (
        time_alloc_chart_input,
        kpis_table_input,
        skip_ship_summary_tab_input,
        skip_ship_vis_input,
        skip_ship_vis_input_prettified,
        skip_ship_vis_prettified_cols,
    ) = get_vis_prerequisites(roles_data, total_fte_capacity)

    time_alloc_chart_rows = []
    for group, metrics in time_alloc_chart_input.items():
        for metric, value in metrics.items():
            time_alloc_chart_rows.append(
                {
                    "Group": group,
                    "Category": metric.replace("_", " ").title(),
                    "Minutes": float(value),
                }
            )

    time_alloc_chart_df = pd.DataFrame(time_alloc_chart_rows)

    # Define custom stacking order (bottom → top)
    kpis_order = ["Time In Store", "Travel Time", "Unallocated Time"]

    agg_df = time_alloc_chart_df.groupby(["Group", "Category"], as_index=False)[
        "Minutes"
    ].sum()

    # Total minutes per group
    agg_df["GroupTotal"] = agg_df.groupby("Group")["Minutes"].transform("sum")

    # Percentage within each group
    agg_df["Percent"] = agg_df["Minutes"] / agg_df["GroupTotal"] * 100

    # Compute stack midpoints for internal labels
    agg_df["y0"] = agg_df.groupby("Group")["Minutes"].cumsum() - agg_df["Minutes"]
    agg_df["y_mid"] = agg_df["y0"] + agg_df["Minutes"] / 2

    # Text for each segment
    agg_df["LabelText"] = agg_df.apply(
        lambda r: f"{r['Minutes']:.0f} ({r['Percent']:.0f}%)", axis=1
    )

    # --- 2️⃣ Base stacked bar chart ---
    kpis_order = ["Time In Store", "Travel Time", "Unallocated Time"]

    bars = (
        alt.Chart(agg_df)
        .mark_bar()
        .encode(
            x=alt.X("Group:N", title=None, axis=alt.Axis(labelAngle=0)).axis(
                labelExpr="[slice(datum.label, 0, 10), slice(datum.label, 10, 20), slice(datum.label, 20, 30), slice(datum.label, 30)]",
                labelAngle=0,
                # labelOffset=-5,
            ),
            y=alt.Y("Minutes:Q", title="Minutes", stack="zero"),
            color=alt.Color(
                "Category:N",
                scale=alt.Scale(scheme="set2", domain=kpis_order),
                legend=alt.Legend(title="Category"),
                sort=kpis_order,
            ),
            order=alt.Order("Category", sort="ascending"),
            tooltip=[
                alt.Tooltip("Group:N"),
                alt.Tooltip("Category:N"),
                alt.Tooltip("Minutes:Q", format=".0f"),
                alt.Tooltip("Percent:Q", format=".1f"),
            ],
        )
        .properties(width=600, height=400, title="Average Time Allocation per Visit")
    )

    # --- 3️⃣ Mid-segment labels (inside bars) ---
    labels = (
        alt.Chart(agg_df)
        .mark_text(fontSize=12, color="black", baseline="middle", align="center")
        .encode(
            x=alt.X("Group:N"),
            y=alt.Y("y_mid:Q"),
            text="LabelText:N",
        )
    )

    # --- 4️⃣ Total labels (on top of bars) ---
    totals_df = (
        agg_df[["Group", "GroupTotal"]]
        .drop_duplicates()
        .assign(y_pos=lambda d: d["GroupTotal"] + 3)  # small offset above bar
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
            x=alt.X("Group:N"),
            y=alt.Y("y_pos:Q"),
            text=alt.Text("GroupTotal:Q", format=".0f"),
        )
    )

    # --- 5️⃣ Combine all layers ---
    time_alloc_chart = bars + labels + total_labels

    kpis_table_input_df = pd.DataFrame(kpis_table_input)

    # Format the values
    def format_value(row_name, val):
        if row_name in ["Utilization (in %)", "Idle time (Incl. Travel / Parking)"]:
            return f"{round(val * 100)}%"  # integer %
        elif type(val) == str:
            return val
        else:
            return f"{val:.1f}"  # one decimal

    kpis_table_input_df_styled = kpis_table_input_df.apply(
        lambda col: [
            format_value(idx, val) for idx, val in zip(kpis_table_input_df.index, col)
        ],
        axis=0,
    )
    kpis_table_input_df_styled.index.name = "Performance indicator"

    kpis_table_input_df_styled = kpis_table_input_df_styled.reset_index()

    skip_ship_summary_tab_input_df = pd.DataFrame(skip_ship_summary_tab_input)

    skip_ship_summary_tab_input_df_styled = skip_ship_summary_tab_input_df.apply(
        lambda col: [
            format_value(idx, val)
            for idx, val in zip(skip_ship_summary_tab_input_df.index, col)
        ],
        axis=0,
    )
    # skip_ship_summary_tab_input_df_styled = skip_ship_summary_tab_input_df.copy()

    cols = st.columns([1, 1])
    with cols[0]:
        st.altair_chart(time_alloc_chart, use_container_width=True)
    with cols[1]:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.dataframe(
            kpis_table_input_df_styled.style.set_properties(**{"text-align": "center"}),
        )

    time_alloc_gran_charts = []
    for key, skip_ship_vis_input_curr in skip_ship_vis_input.items():
        if skip_ship_vis_input_curr.shape[0] > 0:
            # Aggregate Visit Duration per Sales Rep
            skip_ship_vis_input_curr_agg = (
                skip_ship_vis_input_curr.groupby("Sales Rep name")
                .agg({"Visit Duration": "sum"})
                .reset_index()
            )
            skip_ship_vis_input_curr_agg["Visit Duration"] *= 60

            # Sort by Visit Duration descending
            skip_ship_vis_input_curr_agg = skip_ship_vis_input_curr_agg.sort_values(
                "Visit Duration", ascending=False
            )
            skip_ship_vis_input_curr_agg["Sales Rep name"] = pd.Categorical(
                skip_ship_vis_input_curr_agg["Sales Rep name"],
                categories=skip_ship_vis_input_curr_agg["Sales Rep name"],
                ordered=True,
            )

            # Create Altair bar chart with title
            time_alloc_gran_chart = (
                alt.Chart(skip_ship_vis_input_curr_agg)
                .mark_bar(size=10)
                .encode(
                    x=alt.X("Sales Rep name:N", sort=None, title="Sales Rep"),
                    y=alt.Y("Visit Duration:Q", title="Visit Duration"),  # shared scale
                )
                .properties(title=key, height=80)
            )

            time_alloc_gran_charts.append(time_alloc_gran_chart)

    st.subheader("Unallocated visits")

    with st.expander("ℹ️ Unallocated visits explanation", expanded=True):
        st.markdown(
            """
            <div style="
                background-color:#eaf4fe;
                padding:1em 1em 1em 1.5em;
                border-left: 6px solid #1f77b4;
                border-radius:0.5em;
                line-height: 1.5;
            ">
                <b>Unallocated visits</b> are individual visits which Google Maps Route Optimizer <b>was not able to allocate</b> to a given Sales Rep.<br><br>
                This can happen if the visit duration is too long to fit into the working hours of a sales rep.<br><br>
                To <b>mitigate this</b>, you should consider, adjusting below and reruning optimizer:
                <ul>
                    <li>Increasing the number of sales reps (FTEs) in the respective group by manually adjusting the <b>"Number of FTEs"</b> prior to running the Regionalization Engine</li>
                    <li>Reducing time in store for specific customers</li>
                    <li>Increasing the Delivery Days window for a given role and rerunning Week Distribution and Route Optimization</li>
                    <li>Consider updating possible Delivery Days for the specific customers in "Add & Manage Scenarios"</li>
                </ul>
                In case unallocated visits remain, you can manually assign them to the routing results.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.dataframe(
        skip_ship_summary_tab_input_df_styled.style.set_properties(
            **{"text-align": "center"}
        )
    )

    if any([df_temp.shape[0] for df_temp in skip_ship_vis_input_prettified.values()]):
        skip_ship_vis_input_prettified_all = pd.concat(
            skip_ship_vis_input_prettified.values()
        )
    else:
        skip_ship_vis_input_prettified_all = pd.DataFrame(
            columns=skip_ship_vis_prettified_cols
        )

    stores_metadata_cols = ["Stores", "Chain", "City", "Street"]
    stores_sel = stores[["Customer"] + stores_metadata_cols]
    skip_ship_vis_input_prettified_all = pd.merge(
        skip_ship_vis_input_prettified_all,
        stores_sel,
        how="left",
        on="Customer",
        validate="m:1",
    )
    skip_ship_vis_input_prettified_all = skip_ship_vis_input_prettified_all[
        ["Customer"]
        + stores_metadata_cols
        + [c for c in skip_ship_vis_prettified_cols if c != "Customer"]
    ]

    skip_ship_vis_input_prettified_all = skip_ship_vis_input_prettified_all.rename(
        columns={"Distribution Days": "Delivery Days"}
    )

    skip_ship_vis_input_prettified_all = skip_ship_vis_input_prettified_all.reset_index(
        drop=True
    )
    st.dataframe(
        skip_ship_vis_input_prettified_all.style.set_properties(
            **{"text-align": "center"}
        ).format({"Visit Duration (in minutes)": "{:.1f}"}),
    )
    download_dataframe(
        df=skip_ship_vis_input_prettified_all,
        output_file_name="unallocated_visits_detailed",
        button_label="Download Unallocated Visits",
    )

    st.markdown('<h4 style="font-size:16px;"></h4>', unsafe_allow_html=True)

    visits_gran_lst = [roles_data[role]["routes_simple"] for role in roles_data.keys()]

    for role in roles_data.keys():
        visits_gran_curr = roles_data[role]["routes_simple"]
        visits_gran_curr["Role"] = role

    visits_gran_df = pd.concat(visits_gran_lst)

    visits_gran_df = pd.merge(
        visits_gran_df,
        stores_sel,
        how="left",
        on="Customer",
        validate="m:1",
    )

    visits_gran_df = visits_gran_df[
        [
            "Role",
            "Sales Rep name",
            "Week",
            "Weekday",
            "Customer",
            "Stores",
            "Chain",
            "City",
            "Street",
            "Visit Start Time",
            "Visit End time",
            "Travel to Store (min)",
            "Park and Walk (min)",
            "Time in Store (min)",
            "Break Time (min)",
            "Travel to Store (km)",
        ]
    ]
    visits_gran_df = visits_gran_df.sort_values(
        [
            "Role",
            "Sales Rep name",
            "Visit Start Time",
        ]
    )

    st.subheader("Route Optimizer Final Results")

    visits_gran_filt_df = filter_dataframe(df=visits_gran_df, enable_filters=True)
    st.dataframe(visits_gran_filt_df)
    download_dataframe(
        df=visits_gran_df,
        output_file_name="route_optimizer_final_results",
        button_label="Download Route Optimizer Final Results",
    )
