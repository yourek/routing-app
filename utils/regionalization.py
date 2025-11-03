import numpy as np
import pandas as pd
import json
import folium
import geopandas as gpd
from scipy.stats import mode
from k_means_constrained import KMeansConstrained
import streamlit as st
from datetime import datetime
import warnings

from params.regionalization import (
    MONTH_TO_FOUR_WEEKS_ADJUSTER,
    SECONDS_IN_HOUR,
    WEEKS_IN_MONTH,
    BASE_DATE,
)
from utils.models_common import expanded_data, map_workweek_days

from utils.Time_In_Store_Estimate import *


def get_clustering_prerequisites(
    role_name,
    stores,
    scenario,
    project_params,
):
    hours_column = f"{role_name} - time in store"
    freq_column = f"{role_name} - visits per month"

    stores_group = stores.loc[stores[hours_column] > 0].reset_index(drop=True).copy()
    stores_group.loc[:, hours_column] *= MONTH_TO_FOUR_WEEKS_ADJUSTER

    missing_coords = stores_group[["Latitude", "Longitude"]].isna().any(axis=1)
    stores_group = stores_group[~missing_coords].reset_index(drop=True)

    missing_coords_nb = missing_coords.sum()
    if missing_coords_nb:
        st.info(
            f"Removing {missing_coords_nb} of rows for {role_name} due to missing coordinates!"
        )

    if not stores_group.shape[0]:
        st.info(
            f"Skipping regionalization for {role_name}! No rows with coordinates present and time in store more than 0."
        )
        return None

    # 2. Estimation of clusters
    workday_start = datetime.strptime(project_params["work_start"], "%H:%M").time()
    workday_end = datetime.strptime(project_params["work_end"], "%H:%M").time()
    workday_duration = (
        datetime.combine(BASE_DATE, workday_end)
        - datetime.combine(BASE_DATE, workday_start)
    ).total_seconds()

    workweek_days = map_workweek_days(project_params["week_schedule"])
    days_in_workweek = len(workweek_days)

    max_month_hours = int(
        (workday_duration / SECONDS_IN_HOUR) * days_in_workweek * WEEKS_IN_MONTH
    )
    max_month_store_hours = int(round(max_month_hours * scenario.time_in_store / 100))
    min_month_store_hours = max_month_store_hours - 12

    print(
        "Working on...\n",
        "Regionalization",
        "\nMax # total hours in month:\t",
        max_month_hours,
        "\nMax # stores hours in month:\t",
        max_month_store_hours,
        "\nMin # stores hours in month:\t",
        min_month_store_hours,
    )

    return (
        stores_group,
        hours_column,
        freq_column,
        max_month_hours,
        max_month_store_hours,
        min_month_store_hours,
    )


def estimate_n_clusters(
    stores_group,
    hours_column,
    role_name,
    max_month_hours,
    max_month_store_hours,
    store_time_share,
    min_month_store_hours=None,
):
    hours_accuracy = 10

    X = stores_group[["Latitude", "Longitude"]]
    X_expanded, _ = expanded_data(X, stores_group[hours_column], hours_accuracy)

    min_month_store_hours *= hours_accuracy
    max_month_store_hours *= hours_accuracy

    suggested_clusters = float(X_expanded.shape[0] / max_month_store_hours)
    n_clusters = int(np.ceil(suggested_clusters))
    print(
        "Estimated # FTEs:\t",
        round(suggested_clusters, 2),
        "\nSuggested # FTEs:\t",
        n_clusters,
    )

    return n_clusters


def make_weighted_clusters(
    stores_group,
    hours_column,
    role_name,
    n_clusters,
    max_month_store_hours,
    min_month_store_hours=None,
):
    # This is need for handling floating point hours. Accuracy to 0.1h requires
    # parameter to be 10, 0.01 parameter to be 100 etc. The higher the number the
    # longer it takes to run - it basically duplicate points that amount of times.
    hours_accuracy = 10

    X = stores_group[["Latitude", "Longitude"]]
    X_expanded, mapping = expanded_data(X, stores_group[hours_column], hours_accuracy)

    min_month_store_hours *= hours_accuracy
    max_month_store_hours *= hours_accuracy

    if X_expanded.shape[0] <= max_month_store_hours:
        max_month_store_hours = None
        min_month_store_hours = None

    model_fitted = False
    zero_reached = False
    while not model_fitted and not zero_reached:
        if not min_month_store_hours:
            zero_reached = True
            min_month_store_hours = None
        kmeans = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=min_month_store_hours,
            size_max=max_month_store_hours,
            random_state=42,
            verbose=False,
        )
        try:
            labels_expanded = kmeans.fit_predict(X_expanded)
            model_fitted = True
        except ValueError as e:
            min_month_store_hours -= 10
            min_month_store_hours = max(min_month_store_hours, 0)

    clusters = pd.DataFrame({"index": mapping, "sales_rep_group": labels_expanded})
    clusters = clusters.groupby("index").agg({"sales_rep_group": mode})
    clusters["sales_rep_group"] = (
        clusters["sales_rep_group"].str[0].apply(lambda x: f"{1+x:02}_{role_name}")
    )

    stores_group = stores_group.merge(
        clusters, how="left", left_index=True, right_index=True
    )
    cluster_monthly_hours = stores_group.groupby("sales_rep_group").agg(
        {hours_column: "sum"}
    )
    return cluster_monthly_hours, stores_group


def add_visit_duration(stores, hours_column, freq_column):
    stores["Sales Rep name"] = stores["sales_rep_group"]
    stores["Visit Duration"] = stores[hours_column] / stores[freq_column]
    return stores


def create_clusters_map(stores, hours_column):
    cluster_polygons = stores.groupby("sales_rep_group")["geometry"].apply(
        lambda x: x.union_all().convex_hull
    )

    cluster_gdf = gpd.GeoDataFrame(
        cluster_polygons, geometry="geometry", crs=stores.crs
    ).reset_index()
    cluster_gdf["geometry"] = (
        cluster_gdf["geometry"].to_crs("EPSG:2039").buffer(1000).to_crs("EPSG:4326")
    )

    m = cluster_gdf.explore(
        tiles="cartodb positron",
        column="sales_rep_group",
        popup=False,
        tooltip=False,
        legend=False,
    )
    return stores.assign(MonthHours=lambda df: df[hours_column].round(2))[
        [
            "Customer",
            "Stores",
            "City",
            "Street",
            "Chain",
            "Channel",
            "Sales Rep name",
            "MonthHours",
            "geometry",
        ]
    ].explore(
        m=m,
        tiles="cartodb positron",
        column="Sales Rep name",
        popup=True,
        marker_kwds={"radius": 5, "color": "transparent"},
    )


def prepare_legend(rep_color_map):
    legend_html = """
    <div style="
        position: fixed;
        bottom: 15px;
        right: 15px;
        width: 185px;
        background-color: white;
        border: 2px solid grey;
        border-radius: 5px;
        padding: 8px 10px;
        font-size: 12px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        z-index: 9999;
        ">
        <b style="font-size: 13px;">Sales Rep name</b><br>
    """
    for rep, color in rep_color_map.items():
        legend_html += f"""
        <i style="
            background:{color};
            width: 12px;
            height: 12px;
            display: inline-block;
            margin-right: 6px;
            border-radius: 2px;
        "></i>{rep}<br>"""
    legend_html += "</div>"

    return legend_html


def create_scenario_clusters_map(stores_group):
    cluster_polygons = stores_group.groupby("sales_rep_group")["geometry"].apply(
        lambda x: x.union_all().convex_hull
    )

    cluster_gdf = gpd.GeoDataFrame(
        cluster_polygons, geometry="geometry", crs=stores_group.crs
    ).reset_index()
    cluster_gdf["geometry"] = (
        cluster_gdf["geometry"].to_crs("EPSG:2039").buffer(1000).to_crs("EPSG:4326")
    )
    rep_color_map = {
        "Junior": "#9b59b6",
        "Senior": "#3498db",
        "Merchandiser": "#16a085",
    }

    m = folium.Map(
        [
            stores_group["geometry_location_lat"].mean(),
            stores_group["geometry_location_lng"].mean(),
        ],
        zoom_start=10,
        tiles="cartodb positron",
    )
    for sales_rep in sorted(stores_group["sales_rep_group"].unique()):
        if "junior" in sales_rep:
            color = rep_color_map["Junior"]
        elif "senior" in sales_rep:
            color = rep_color_map["Senior"]
        elif "merch" in sales_rep:
            color = rep_color_map["Merchandiser"]

        fg = folium.FeatureGroup(name=sales_rep, show=True).add_to(m)
        (
            cluster_gdf.loc[cluster_gdf["sales_rep_group"].eq(sales_rep)].explore(
                m=fg,
                tiles="cartodb positron",
                color=color,
                popup=True,
                tooltip=False,
                legend=False,
            )
        )

    stores_group["sales_rep_class"] = stores_group["sales_rep_group"].apply(
        lambda x: (
            "Junior"
            if "junior" in x
            else ("Senior" if "senior" in x else "Merchandiser")
        )
    )
    for sales_rep_class in sorted(stores_group["sales_rep_class"].unique()):
        fg = folium.FeatureGroup(name=f"{sales_rep_class} Stores", show=True).add_to(m)
        (
            stores_group.loc[stores_group["sales_rep_class"].eq(sales_rep_class)][
                [
                    "Customer",
                    "Stores",
                    "City",
                    "Street",
                    "Chain",
                    "Channel",
                    "Sales Rep name",
                    "geometry",
                ]
            ].explore(
                m=fg,
                tiles="cartodb positron",
                color=rep_color_map[sales_rep_class],
                popup=True,
                marker_kwds={"radius": 5, "color": "transparent"},
            )
        )

    folium.LayerControl(position="bottomleft").add_to(m)
    folium.FitOverlays().add_to(m)

    legend_html = prepare_legend(rep_color_map)
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def prepare_groups_data(role_name, output_dir):
    stores_group_geo = gpd.read_parquet(
        output_dir / "intermediate" / f"{role_name}_stores_group.parquet"
    )
    stores_group = pd.read_excel(
        output_dir / f"stores_group.xlsx",
        sheet_name=f"Data ({role_name})",
        dtype=stores_group_geo.dtypes.to_dict(),
    )
    stores_group["sales_rep_group"] = stores_group["Sales Rep name"]

    stores_group = gpd.GeoDataFrame(stores_group, geometry=stores_group_geo["geometry"])
    return stores_group


def save_clustering_results(results, scenario_path):
    rgn_res_folder = scenario_path / "rgn_res"
    rgn_res_folder.mkdir(parents=True, exist_ok=True)

    for result in results:
        role_name = result["role_name"]
        result_ = result["stores_group"].drop(columns="geometry")
        with open(rgn_res_folder / f"{role_name}.json", "w", encoding="utf-8") as f:
            json.dump(
                result_.to_dict(orient="records"), f, ensure_ascii=False, indent=2
            )


def estimate_n_clusters_all(scenario_folder, scenario):
    # 1. Data preparation
    scenario_json_path = scenario_folder / scenario.id / "scenario_metadata.json"
    input_output_path = scenario_folder / scenario.id / "stores_w_role_spec.json"
    project_params_path = scenario_folder.parent / "input_parameters.json"
    with open(scenario_json_path, "r") as f:
        scen_metadata = json.load(f)
    with open(input_output_path, "r", encoding="utf-8") as f:
        stores_json = json.load(f)
        stores = pd.DataFrame(stores_json)
    with open(project_params_path, "r", encoding="utf-8") as f:
        project_params = json.load(f)

    stores = gpd.GeoDataFrame(
        stores.reset_index(drop=True),
        geometry=gpd.points_from_xy(
            stores["Longitude"],
            stores["Latitude"],
            crs="EPSG:4326",
        ),
    )

    info_placeholder = st.empty()

    # Kris
    if scenario.time_in_store == 100:
        info_placeholder.info(
            "**Time in Store factor** not provided. Running estimation...",
            icon="ℹ️",
            width="stretch",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            time_estimations, _ = TimeInStoreEst(
                stores_json,
                min_cluster_size=100,
                clusters_sampled=1,
                lunch_brake_in_minutes=project_params["lunch_break"],
                in_out_penalty_minutes=project_params["parking_delay"],
                turn_penalty_seconds=6,
            )

            if time_estimations is not None:

                float_time = np.mean(
                    sum(
                        [
                            [time_estimations[k][y] for y in time_estimations[k]]
                            for k in time_estimations
                        ],
                        [],
                    )
                )

                new_value = int(round(100 * float_time, 0))
                # store in scenario
                scenario.time_in_store = new_value
                scenario.save(scenario_folder)

                info_placeholder.success(
                    "Time in Store Calculated: " + str(new_value) + "%",
                    icon="✅",
                    width="stretch",
                )

    n_clusters_dict = dict()
    role_names = [role["name"] for role in scen_metadata["roles"]]
    for role_name in role_names:
        if (
            res := get_clustering_prerequisites(
                role_name,
                stores,
                scenario,
                project_params,
            )
        ) is not None:
            (
                stores_group,
                hours_column,
                freq_column,
                max_month_hours,
                max_month_store_hours,
                min_month_store_hours,
            ) = res
        else:
            continue

        n_clusters = estimate_n_clusters(
            stores_group,
            hours_column,
            role_name,
            max_month_hours,
            max_month_store_hours,
            scenario.time_in_store,
            min_month_store_hours,
        )
        n_clusters_dict[role_name] = n_clusters

    return n_clusters_dict


def regionalization(scenario_folder, scenario, n_clusters_dict):
    # 1. Data preparation
    scenario_json_path = scenario_folder / scenario.id / "scenario_metadata.json"
    input_output_path = scenario_folder / scenario.id / "stores_w_role_spec.json"
    project_params_path = scenario_folder.parent / "input_parameters.json"
    with open(scenario_json_path, "r") as f:
        scen_metadata = json.load(f)
    with open(input_output_path, "r", encoding="utf-8") as f:
        stores_json = json.load(f)
        stores = pd.DataFrame(stores_json)
    with open(project_params_path, "r", encoding="utf-8") as f:
        project_params = json.load(f)

    stores = gpd.GeoDataFrame(
        stores.reset_index(drop=True),
        geometry=gpd.points_from_xy(
            stores["Longitude"],
            stores["Latitude"],
            crs="EPSG:4326",
        ),
    )

    info_placeholder = st.empty()

    info_placeholder.info("Running regionalization engine..", icon="ℹ️", width="stretch")

    results = list()
    role_names = [role["name"] for role in scen_metadata["roles"]]
    for role_name in role_names:
        info_placeholder.info(f"Working on {role_name}...", icon="ℹ️", width="stretch")

        if (
            res := get_clustering_prerequisites(
                role_name,
                stores,
                scenario,
                project_params,
            )
        ) is not None:
            (
                stores_group,
                hours_column,
                freq_column,
                max_month_hours,
                max_month_store_hours,
                min_month_store_hours,
            ) = res
        else:
            continue

        # 3. Clustering main
        _, stores_group = make_weighted_clusters(
            stores_group=stores_group,
            hours_column=hours_column,
            role_name=role_name,
            n_clusters=n_clusters_dict[role_name],
            max_month_store_hours=max_month_store_hours,
            min_month_store_hours=min_month_store_hours,
        )

        stores_group = add_visit_duration(stores_group, hours_column, freq_column)
        m_clusters = create_clusters_map(stores_group, hours_column)

        stores_group = stores_group.drop(columns="sales_rep_group")

        results.append(
            {
                "role_name": role_name,
                "stores_group": stores_group,
                "m_clusters": m_clusters,
            }
        )

    save_clustering_results(results, scenario_folder / scenario.id)
    scenario.regionalization_completed = True
    scenario.time_in_store_used_in_rgn = scenario.time_in_store
    scenario.save(scenario_folder)

    return "Regionalization completed successfully!"
