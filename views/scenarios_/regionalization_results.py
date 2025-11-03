import json
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import pandas as pd
import textwrap
import pydeck as pdk
import streamlit as st
from shapely.geometry import MultiPoint, Polygon, MultiPolygon
import alphashape
from datetime import datetime
import random

from utils.data_loaders import load_default_input_parameters
from utils.dataframe_utils import (
    download_dataframe,
    filter_dataframe,
    render_editable_table,
)
from utils.models_common import get_max_month_store_hours


def get_legend_html(color_map, alpha):
    legend_html_beginning = """
    <div style="
        z-index: 1000;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 12px 18px;
        border-radius: 8px;
        box-shadow: 0px 0px 8px rgba(0,0,0,0.2);
        font-size: 14px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px 16px;
    ">
    <b style="width: 100%;">Legend</b>
    """
    legend_html_sales_rep_template = """
    <div style="display: flex; align-items: center; gap: 6px;">
        <div style="width: 15px; height: 15px; background: rgba({color}); border-radius: 2px;"></div>
        {sales_rep_name}
    </div>
    """
    legend_html_ending = "</div>"

    legend_html = legend_html_beginning
    for sales_rep, color in color_map.items():
        legend_html += legend_html_sales_rep_template.format(
            sales_rep_name=sales_rep, color=",".join(map(str, color[:3] + [alpha]))
        )
    legend_html += legend_html_ending

    return legend_html


def build_polygons(rgn_res, alpha=0.3, buffer_km=10):
    """
    Build polygons (territories) around each Sales Rep's points.
    Uses alpha shape (concave hull) when possible, otherwise convex hull.
    Fallback: buffer around points for 1–2 locations.
    """
    polygons = []
    buffer_deg = buffer_km / 111.0  # approx conversion: 1° ≈ 111 km

    for rep, group in rgn_res.groupby("Sales Rep name"):
        points = list(zip(group["Longitude"], group["Latitude"]))

        # Alpha shape if enough points
        if len(points) >= 4:
            poly = alphashape.alphashape(points, alpha)
        elif len(points) >= 3:
            poly = MultiPoint(points).convex_hull
        else:
            poly = MultiPoint(points).buffer(buffer_deg)

        # Handle MultiPolygon → take largest
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda a: a.area)

        # Only polygons with area > 0
        if isinstance(poly, Polygon) and not poly.is_empty:
            polygons.append(
                {"Sales Rep name": rep, "coordinates": list(poly.exterior.coords)}
            )

    return pd.DataFrame(polygons)


def plot_rgn_res_on_map(rgn_res, alpha=0.3, buffer_km=10):
    rgn_res = rgn_res.copy()

    # Assign unique RGBA colors (semi-transparent)
    unique_reps = rgn_res["Sales Rep name"].sort_values().unique()

    random.seed(42)
    color_map = {
        rep: [
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
            int(alpha * 255),
        ]
        for rep in unique_reps
    }

    rgn_res["color"] = rgn_res["Sales Rep name"].map(color_map)

    # Build polygons
    polygons = build_polygons(rgn_res, alpha=alpha, buffer_km=buffer_km)
    polygons["color"] = polygons["Sales Rep name"].map(color_map)

    rgn_res["Tooltip"] = rgn_res.apply(
        lambda row: f"""
        <b>Customer</b>: {row['Customer']}<br>
        <b>Store</b>: {row['Stores']}<br>
        <b>Chain</b>: {row['Chain']}<br>
        <b>City</b>: {row['City']}<br>
        <b>Street</b>: {row['Street']}<br>
        <b>Channel</b>: {row['Channel']}<br>
        <b>Group</b>: {row['Group']}<br>
        <b>Grading</b>: {row['Grading']}<br>
        <b>Sales Avg.</b>: {row['Sales Avg']}<br>
        """,
        axis=1,
    )
    polygons["Tooltip"] = polygons.apply(lambda row: row["Sales Rep name"], axis=1)

    # Polygon layer (behind)
    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=polygons,
        get_polygon="coordinates",
        get_fill_color="color",
        get_line_color="color",
        line_width_min_pixels=2,
        pickable=True,  # polygons don’t need tooltips
    )

    # Scatterplot layer (on top)
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=rgn_res,
        get_position="[Longitude, Latitude]",
        get_fill_color="color",
        get_radius=200,
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": ("{Tooltip}"),
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }

    # Map view
    view_state = pdk.ViewState(
        latitude=rgn_res["Latitude"].mean(),
        longitude=rgn_res["Longitude"].mean(),
        zoom=7,
        pitch=0,
    )

    # Deck map
    rgn_map = pdk.Deck(
        layers=[polygon_layer, scatter_layer],
        initial_view_state=view_state,
        map_style="light",
        tooltip=tooltip,
    )

    st.pydeck_chart(rgn_map)

    legend_html = """
    <div style="
        position: absolute;
        bottom: 40px;
        left: 40px;
        z-index: 1000;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 12px 18px;
        border-radius: 8px;
        box-shadow: 0px 0px 8px rgba(0,0,0,0.2);
        font-size: 14px;
    ">
    <b>Legend</b><br>
    <div style="display: flex; align-items: center; gap: 6px;">
    <div style="width: 15px; height: 15px; background: rgba(255,0,0,0.7); border-radius: 2px;"></div>
    Region A
    </div>
    <div style="display: flex; align-items: center; gap: 6px;">
    <div style="width: 15px; height: 15px; background: rgba(0,200,0,0.7); border-radius: 50%;"></div>
    Point Type A
    </div>
    </div>
    """

    legend_html = get_legend_html(color_map, alpha)

    st.markdown(legend_html, unsafe_allow_html=True)


def draw_regionalization_roles_cards(roles_data, max_capacity_per_fte):
    role_names = list(roles_data.keys())
    role_names_sorted = sorted(role_names, key=str.lower)

    cols = st.columns(len(role_names_sorted))

    for col, role_name in zip(cols, role_names_sorted):
        rgn_res_curr = roles_data[role_name]
        time_in_store = rgn_res_curr[f"{role_name} - time in store"].sum()
        fte_nb = rgn_res_curr["Sales Rep name"].nunique()
        time_in_store_per_fte = time_in_store / fte_nb

        with col:
            st.markdown(
                f"""
                <div style="background-color:#f8f9fa;
                            padding:1rem;
                            border-radius:0.75rem;
                            box-shadow:0 2px 4px rgba(0,0,0,0.1);
                            text-align:center;">
                    <h4 style="margin-bottom:0.5rem;">{role_name}</h4>
                    <p style="margin:0;"><b>{fte_nb}</b> FTEs</p>
                    <p style="margin:0;">Total time in store: <b>{np.round(time_in_store, 0)}h</b></p>
                    <p style="margin:0;">Avg. time in store a month per FTE: <b>{int(np.round(time_in_store_per_fte, 0))}h</b></p>
                    <p style="margin:0;">Max capacity a month per FTE: <b>{max_capacity_per_fte}h</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def draw_single_role_monthly_hours_chart(
    rgn_res, selected_role, scenario, project_params
):
    metric = f"{selected_role} - time in store"
    rgn_res_agg = rgn_res.groupby("Sales Rep name").agg({metric: "sum"}).reset_index()

    chart = (
        alt.Chart(rgn_res_agg)
        .mark_bar()
        .encode(
            y=alt.Y(
                "Sales Rep name",
                sort="-x",
                axis=alt.Axis(labelAlign="right", labelPadding=10, labelLimit=500),
            ),
            x=f"{selected_role} - time in store",
        )
    )

    text = (
        alt.Chart(rgn_res_agg)
        .mark_text(
            align="left",
            baseline="middle",
            dx=5,
        )
        .encode(
            y=alt.Y("Sales Rep name:N", sort="-x"),
            x=alt.X(f"{metric}:Q"),
            text=alt.Text(f"{metric}:Q", format=".0f"),
        )
    )

    max_month_store_hours = get_max_month_store_hours(project_params, scenario)
    constant_line = (
        alt.Chart(rgn_res_agg)
        .mark_rule(strokeDash=[5, 5], color="red")
        .encode(x=alt.datum(max_month_store_hours))
        .properties()
    )

    chart = (chart + constant_line + text).interactive()
    chart_placeholder = st.empty()
    final_chart_to_render = chart_placeholder.altair_chart(
        chart,
        use_container_width=True,
    )

    return final_chart_to_render


def load_roles_regionaliztion_results(scenario, scenario_folder):
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
                scenario_folder / scenario.id / f"rgn_res/{role_name}.json",
                "r",
                encoding="utf-8",
            ) as f:
                rgn_res_curr = pd.DataFrame(json.load(f))
        except FileNotFoundError:
            st.warning(
                f"Role '{role_name}' has no regionalization results. In order to see results, please re-run the regionalization process."
            )
            continue

        roles_data[role_name] = rgn_res_curr

    return roles_data


def regionalization_results(
    scenario,
    scenario_folder,
):
    if not scenario.regionalization_completed:
        st.info(
            "No regionalization results. Please run the process to see the results."
        )
        return

    roles_data = load_roles_regionaliztion_results(scenario, scenario_folder)

    active_project = st.session_state.get("active_project")
    project_id = active_project["id"]
    input_parameter_dict = load_default_input_parameters(project_id)

    working_hours = input_parameter_dict.get("working_hours", None)
    max_capacity_per_fte = int(
        round(scenario.time_in_store_used_in_rgn * working_hours / 100, 0)
    )

    draw_regionalization_roles_cards(roles_data, max_capacity_per_fte)

    # Single role selector
    role_names_sorted = list(roles_data.keys())
    selected_role = st.selectbox(
        "Select role", role_names_sorted, index=0, help="Choose a role"
    )
    sel_rol_rgn_res_path = (
        scenario_folder / scenario.id / f"rgn_res/{selected_role}.json"
    )
    rgn_res = roles_data[selected_role]
    st.session_state["rgn_res"] = rgn_res

    cols = st.columns([2, 3])
    with cols[0]:
        draw_single_role_monthly_hours_chart(
            st.session_state["rgn_res"], selected_role, scenario, input_parameter_dict
        )
    with cols[1]:
        plot_rgn_res_on_map(st.session_state["rgn_res"])

    static_columns_config = {
        col: st.column_config.Column(
            "\n".join(textwrap.wrap(col, width=10)),
            width="auto",
            disabled=True,
        )
        for col in rgn_res.columns
        if col != "Sales Rep name"
    }

    groups_lst = rgn_res["Sales Rep name"].drop_duplicates().sort_values().to_list()
    dynamic_columns_config = {
        "Sales Rep name": st.column_config.SelectboxColumn(
            "\n".join(textwrap.wrap("Sales Rep name", width=10)),
            width="auto",
            help="Select a region",
            options=groups_lst,
            disabled=False,
        )
    }
    column_config = dynamic_columns_config | static_columns_config

    column_order = ["Sales Rep name"] + [
        c for c in rgn_res.columns if c != "Sales Rep name"
    ]

    filtered_df = filter_dataframe(rgn_res, True)
    render_editable_table(
        df=filtered_df,
        column_config=column_config,
        update_key="Customer",
        data_path=sel_rol_rgn_res_path,
        key_pattern="rgn_res",
        cols_to_hide=[],
        column_order=column_order,
    )

    download_dataframe(
        rgn_res, f"{scenario.name}_{selected_role}_regionalization_results"
    )
