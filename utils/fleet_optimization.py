import json
import geopandas as gpd
import numpy as np
import pandas as pd
from datetime import timedelta
from google.maps import routeoptimization_v1 as ro
from google.protobuf.json_format import MessageToDict, MessageToJson
from tqdm import tqdm
import polyline
from shapely.geometry import LineString, Point
from datetime import datetime


from params.regionalization import (
    WEEKS_IN_MONTH,
    BASE_DATE,
    METERS_IN_KM,
    SECONDS_IN_MINUTE,
    SECONDS_IN_HOUR,
)
from utils.dataframe_utils import convert_date_types_to_string
from utils.models_common import (
    clear_folder,
    map_workweek_days,
    get_dayname_to_daynum_mapping,
)

pd.options.display.max_columns = None


def polyline_to_linestring(s):
    if pd.isna(s):
        return None

    decoded_points = polyline.decode(s)

    if not decoded_points:
        return None
    elif len(decoded_points) == 1:
        lat, lon = decoded_points[0]
        return Point(lon, lat)
    else:
        return LineString([(lon, lat) for lat, lon in decoded_points])


def save_routing_results(results, scenario_path):
    fleet_opt_res_folder = scenario_path / "fleet_opt" / "final"
    fleet_opt_res_folder.mkdir(parents=True, exist_ok=True)

    for result in results:
        role_name = result["role_name"]
        result["routes_simple"] = convert_date_types_to_string(result["routes_simple"])
        result["routes_simple"]["geometry"] = result["routes_simple"][
            "geometry"
        ].astype(str)

        result["skipped_shipments"] = convert_date_types_to_string(
            result["skipped_shipments"]
        )
        if "geometry" in result["skipped_shipments"]:
            result["skipped_shipments"]["geometry"] = result["skipped_shipments"][
                "geometry"
            ].astype(str)

        with open(
            fleet_opt_res_folder / f"{role_name} - routes_simple.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result["routes_simple"].to_dict(orient="records"),
                f,
                ensure_ascii=False,
                indent=2,
            )
        with open(
            fleet_opt_res_folder / f"{role_name} - skipped_shipments.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result["skipped_shipments"].to_dict(orient="records"),
                f,
                ensure_ascii=False,
                indent=2,
            )


def fleet_optimization(
    stores_spread,
    month_schedule,
    stores_group,
    role_name,
    res_dir,
    project_params,
):
    workday_start = datetime.strptime(project_params["work_start"], "%H:%M").time()
    workday_end = datetime.strptime(project_params["work_end"], "%H:%M").time()

    sales_reps = sorted(set(stores_group["Sales Rep name"]))
    freq_column = f"{role_name} - visits per month"

    workweek_days = map_workweek_days(project_params["week_schedule"])
    days_in_workweek = len(workweek_days)

    day_to_num = get_dayname_to_daynum_mapping()

    for sales_rep in tqdm(sales_reps):
        for i in range(WEEKS_IN_MONTH):
            start_date = BASE_DATE + timedelta(weeks=i)
            global_start_time = (
                start_date + timedelta(days=day_to_num[workweek_days[0]])
            ).replace(
                hour=workday_start.hour,
                minute=workday_start.minute,
            ).isoformat() + "Z"
            global_end_time = (
                start_date + timedelta(weeks=i, days=day_to_num[workweek_days[-1]])
            ).replace(
                hour=workday_end.hour, minute=workday_end.minute
            ).isoformat() + "Z"

            week_stores = stores_spread.loc[
                stores_spread["Sales Rep name"].eq(sales_rep)
                & stores_spread["Week"].eq(f"Week{i+1}")
            ].copy()

            week_schedule = month_schedule.loc[
                month_schedule["sales_rep"].eq(sales_rep)
                & month_schedule["week"].eq(f"Week{i+1}")
            ].copy()

            # 1. Input setup
            shipments = []
            for _, row in week_stores.iterrows():
                customer_time_windows = list()
                for di in row["VisitDays"].split(","):
                    di_in_workdays = day_to_num[workweek_days[int(di)]]
                    start_time = (start_date + timedelta(days=di_in_workdays)).replace(
                        hour=workday_start.hour, minute=workday_start.minute
                    )
                    end_time = (start_date + timedelta(days=di_in_workdays)).replace(
                        hour=workday_end.hour, minute=workday_end.minute
                    )

                    customer_time_windows.append(
                        {
                            "start_time": start_time.isoformat() + "Z",
                            "end_time": end_time.isoformat() + "Z",
                        }
                    )

                customer = {
                    "label": row["Customer"],
                    "deliveries": [
                        {
                            "arrival_location": {
                                "latitude": row["Latitude"],
                                "longitude": row["Longitude"],
                            },
                            "duration": f"{int(row["Visit Duration"]*SECONDS_IN_HOUR)}s",
                            "time_windows": customer_time_windows,
                        }
                    ],
                }
                shipments.append(customer)

            vehicles = []
            for _, row in week_schedule.iterrows():
                daily_vehicle = {
                    "label": f"{row['sales_rep']}_{row['week'].lower()}_{row['day_name'].lower()}",
                    "travel_mode": "DRIVING",
                    "cost_per_kilometer": 1.0,
                    "start_time_windows": [
                        {
                            "start_time": row["start"],
                            "end_time": row["end"],
                            # "soft_end_time":  row["soft_end"]
                        }
                    ],
                    "end_time_windows": [
                        {
                            "start_time": row["start"],
                            "end_time": row["end"],
                            # "soft_end_time": row["soft_end"]
                        }
                    ],
                    "break_rule": {
                        "break_requests": [
                            {
                                "earliest_start_time": row["break_time_earliest"],
                                "latest_start_time": row["break_time_latest"],
                                "min_duration": f"{int(project_params['lunch_break'] * SECONDS_IN_MINUTE)}s",
                            }
                        ]
                    },
                }
                vehicles.append(daily_vehicle)

            # 2. API call
            client = ro.RouteOptimizationClient()
            request = ro.OptimizeToursRequest(
                parent="projects/bank-locations-412913",
                model={
                    "shipments": shipments,
                    "vehicles": vehicles,
                    "global_start_time": global_start_time,
                    "global_end_time": global_end_time,
                },
                consider_road_traffic=project_params["road_traffic"],
                populate_polylines=True,
                populate_transition_polylines=True,
            )
            response = client.optimize_tours(request=request)

            # 3. Saving response to the file
            response_dict = MessageToDict(
                response._pb, preserving_proto_field_name=True
            )
            response_json = MessageToJson(
                response._pb, preserving_proto_field_name=True
            )

            response_file_name = f"optimize_tours_{sales_rep.lower()}_week{i+1}.json"
            with open(res_dir / response_file_name, "w") as f:
                f.write(response_json)
            break
        break

    # 6. Gathering Google Maps Fleet Optimization API response files
    response_files = [f for f in list(res_dir.walk())[0][2]]
    response_files

    # 7. Reading Google Maps Fleet Optimization API results
    routes = list()
    metrics = list()
    skipped_shipments = list()
    for response_file in response_files:
        with open(res_dir / response_file) as f:
            response_dict = json.load(f)

        group = response_file.split("_", 2)[2].replace(".json", "")
        rep, week = group.rsplit("_", 1)

        metrics_rep = pd.json_normalize(response_dict["metrics"], sep="_")
        metrics_rep["sales_rep"] = [rep]
        metrics_rep["week"] = [week.title()]
        metrics.append(metrics_rep)

        if "skipped_shipments" in response_dict:
            skipped_shipments_rep = pd.DataFrame(response_dict["skipped_shipments"])
            skipped_shipments_rep["group"] = group
            skipped_shipments.append(skipped_shipments_rep)

        routes_rep = list()
        for i in range(len(response_dict["routes"])):
            try:
                routes_rep_day = pd.concat(
                    [
                        pd.DataFrame(response_dict["routes"][i]["transitions"]),
                        (
                            pd.DataFrame(response_dict["routes"][i]["visits"]).rename(
                                columns={"start_time": "start_time_visit"}
                            )
                        ),
                    ],
                    axis=1,
                )
                routes_rep_day["vehicle_label"] = response_dict["routes"][i][
                    "vehicle_label"
                ]
                routes_rep.append(routes_rep_day)
            except KeyError:
                print(response_file)

        routes_rep = pd.concat(routes_rep, ignore_index=True)

        routes_rep["sales_rep"] = rep
        routes_rep["week"] = week.title()
        routes_rep["weekday"] = (
            routes_rep["vehicle_label"].str.rsplit("_", n=1).str[-1].str.title()
        )
        routes_rep["start_time"] = pd.to_datetime(
            routes_rep["start_time"]
        ).dt.tz_localize(None)
        routes_rep["start_time_visit"] = pd.to_datetime(
            routes_rep["start_time_visit"]
        ).dt.tz_localize(None)

        route_polyline = pd.json_normalize(routes_rep["route_polyline"])
        if route_polyline.empty:
            routes_rep["route_polyline"] = pd.NA
            routes_rep["travel_distance_meters"] = np.nan
            routes_rep["route_token"] = pd.NA
        else:
            routes_rep["route_polyline"] = route_polyline
        routes_rep = gpd.GeoDataFrame(
            (
                routes_rep.drop(
                    columns=["route_polyline", "route_token"]
                ).convert_dtypes()
            ),
            geometry=routes_rep["route_polyline"].apply(polyline_to_linestring),
            crs="EPSG:4326",
        )

        routes.append(routes_rep)

    if len(skipped_shipments) > 0:
        skipped_shipments = pd.concat(skipped_shipments, ignore_index=True)
        skipped_shipments = skipped_shipments.rename(
            columns={"label": "Customer"}
        ).astype({"Customer": "str"})
        visit_hours = stores_spread.copy()
        visit_hours["group"] = (
            visit_hours["Sales Rep name"] + "_" + visit_hours["Week"]
        ).str.lower()
        visit_hours = visit_hours[
            [
                "Customer",
                "group",
                "Distribution Days",
                freq_column,
                "Visit Duration",
                "geometry",
            ]
        ].drop_duplicates()

        skipped_shipments = visit_hours.merge(
            skipped_shipments, how="right", on=["Customer", "group"]
        )
    else:
        skipped_shipments = gpd.GeoDataFrame()

    metrics = pd.concat(metrics, ignore_index=True)
    routes = pd.concat(routes, ignore_index=True)
    routes = routes.dropna(subset="shipment_label").reset_index(drop=True)

    # 8. Standardizing API results
    routes = stores_group.drop(columns="geometry").merge(
        routes, how="inner", left_on="Customer", right_on="shipment_label"
    )
    routes = gpd.GeoDataFrame(routes, geometry=routes["geometry"])

    single_point_routes_map = (
        stores_group.loc[
            stores_group["Customer"].isin(
                routes.loc[routes["geometry"].isna(), "Customer"]
            ),
            ["Customer", "geometry"],
        ]
        .set_index("Customer")
        .to_dict(orient="dict")["geometry"]
    )
    routes["geometry"] = routes.apply(
        lambda r: (
            single_point_routes_map[r["Customer"]]
            if pd.isna(r["geometry"])
            else r["geometry"]
        ),
        axis=1,
    )

    routes["travel_to_store_sec"] = (
        routes["travel_duration"].fillna("0s").str.rstrip("s").astype(int)
    )
    routes["park_walk_sec"] = project_params["parking_delay"] * SECONDS_IN_MINUTE
    routes.loc[routes["travel_to_store_sec"].eq(0), "park_walk_sec"] = 0

    routes["other_idles_sec"] = 0
    for idle in project_params["other_idle_times"]:
        if idle["applicable_cities"]:
            routes.loc[
                routes["City"].isin(idle["applicable_cities"]), "other_idles_sec"
            ] += (idle["minutes"] * SECONDS_IN_MINUTE)
        else:
            routes.loc[:, "other_idles_sec"] += idle["minutes"] * SECONDS_IN_MINUTE
    routes.loc[routes["travel_to_store_sec"].eq(0), "other_idles_sec"] = 0

    routes["break_sec"] = (
        routes["break_duration"].fillna("0s").str.rstrip("s").astype(int)
    )
    routes["detour_sec"] = routes["detour"].fillna("0s").str.rstrip("s").astype(int)
    routes["time_in_store_sec"] = (routes["Visit Duration"] * SECONDS_IN_HOUR).astype(
        int
    )

    routes = routes.sort_values(["sales_rep", "start_time_visit"]).reset_index(
        drop=True
    )

    # 9. Finalizing output structure
    routes_simple = list()
    for sales_rep in routes["sales_rep"].unique():

        for week_n in range(WEEKS_IN_MONTH):
            week = f"Week{week_n+1}"

            for weekday in workweek_days:

                routes_rep = (
                    routes.loc[
                        routes["sales_rep"].eq(sales_rep)
                        & routes["week"].eq(week)
                        & routes["weekday"].eq(weekday)
                    ]
                    .reset_index(drop=True)
                    .copy()
                )
                if routes_rep.empty:
                    continue

                routes_rep.loc[0, "visit_start_time"] = routes_rep.loc[
                    0, "start_time"
                ].replace(
                    hour=workday_start.hour, minute=workday_start.minute, second=0
                )
                routes_rep.loc[0, "visit_end_time"] = routes_rep.loc[
                    0, "visit_start_time"
                ] + timedelta(seconds=int(routes_rep.loc[0, "time_in_store_sec"]))
                for i in range(1, routes_rep.shape[0]):
                    routes_rep.loc[i, "visit_start_time"] = routes_rep.loc[
                        i - 1, "visit_end_time"
                    ] + timedelta(
                        seconds=int(
                            routes_rep.loc[i, "travel_to_store_sec"]
                            + routes_rep.loc[i, "park_walk_sec"]
                            + routes_rep.loc[i, "other_idles_sec"]
                            + routes_rep.loc[i, "break_sec"]
                        )
                    )
                    routes_rep.loc[i, "visit_end_time"] = routes_rep.loc[
                        i, "visit_start_time"
                    ] + timedelta(seconds=int(routes_rep.loc[i, "time_in_store_sec"]))

                routes_simple.append(routes_rep)

    routes_simple = pd.concat(routes_simple, ignore_index=True)
    routes_simple["Sales Rep name"] = routes_simple["sales_rep"]

    routes_simple["travel_to_store_km"] = (
        routes_simple["travel_distance_meters"].fillna(0) / METERS_IN_KM
    )
    routes_simple["travel_to_store_min"] = (
        routes_simple["travel_to_store_sec"] / SECONDS_IN_MINUTE
    ).round(2)
    routes_simple["park_walk_min"] = (
        routes_simple["park_walk_sec"] / SECONDS_IN_MINUTE
    ).round(2)
    routes_simple["other_idles_min"] = (
        routes_simple["other_idles_sec"] / SECONDS_IN_MINUTE
    ).round(2)
    routes_simple["time_in_store_min"] = (
        routes_simple["time_in_store_sec"] / SECONDS_IN_MINUTE
    ).round(2)
    routes_simple["break_time_min"] = (
        routes_simple["break_sec"] / SECONDS_IN_MINUTE
    ).round(2)

    # TODO: investigate
    routes_simple = gpd.GeoDataFrame(
        routes_simple[
            [
                "Sales Rep name",
                "Customer",
                # "Store",
                # "Chain English",
                "travel_to_store_km",
                "travel_to_store_min",
                "park_walk_min",
                "other_idles_min",
                "time_in_store_min",
                "break_time_min",
                freq_column,
                # "Merchandising",
                # "Distribution Days",
                "week",
                "weekday",
                "visit_start_time",
                "visit_end_time",
            ]
        ],
        geometry=routes_simple["geometry"],
    )
    routes_simple = routes_simple.rename(
        columns={
            "travel_to_store_km": "Travel to Store (km)",
            "travel_to_store_min": "Travel to Store (min)",
            "park_walk_min": "Park and Walk (min)",
            "other_idles_min": "Other Idles (min)",
            "time_in_store_min": "Time in Store (min)",
            "break_time_min": "Break Time (min)",
            freq_column: "Monthly Visit Frequency",
            "week": "Week",
            "weekday": "Weekday",
            "visit_start_time": "Visit Start Time",
            "visit_end_time": "Visit End time",
        }
    )
    return routes_simple, skipped_shipments


def fleet_optimization_all(scenario_folder, scenario):
    scenario_json_path = scenario_folder / scenario.id / "scenario_metadata.json"
    project_params_path = scenario_folder.parent / "input_parameters.json"
    res_dir = scenario_folder / scenario.id / "fleet_opt"
    res_dir.mkdir(parents=True, exist_ok=True)

    with open(scenario_json_path, "r") as f:
        scen_metadata = json.load(f)
    with open(project_params_path, "r", encoding="utf-8") as f:
        project_params = json.load(f)

    results = list()
    for role in scen_metadata["roles"]:
        role_name = role["name"]

        try:
            with open(
                scenario_folder / scenario.id / "rgn_res" / f"{role_name}.json",
                "r",
                encoding="utf-8",
            ) as f:
                stores_group = pd.DataFrame(json.load(f))
            with open(
                scenario_folder
                / scenario.id
                / "week_dist_res"
                / f"{role_name} - stores_spread.json",
                "r",
                encoding="utf-8",
            ) as f:
                stores_spread = pd.DataFrame(json.load(f))
            with open(
                scenario_folder
                / scenario.id
                / "week_dist_res"
                / f"{role_name} - month_schedule.json",
                "r",
                encoding="utf-8",
            ) as f:
                month_schedule = pd.DataFrame(json.load(f))
        except FileNotFoundError:
            continue

        geometry = [
            Point(xy) for xy in zip(stores_group["Longitude"], stores_group["Latitude"])
        ]
        stores_group = gpd.GeoDataFrame(stores_group, geometry=geometry)
        stores_group.set_crs(epsg=4326, inplace=True)

        geometry = [
            Point(xy)
            for xy in zip(stores_spread["Longitude"], stores_spread["Latitude"])
        ]
        stores_spread = gpd.GeoDataFrame(stores_spread, geometry=geometry)
        stores_spread.set_crs(epsg=4326, inplace=True)

        res_dir_curr = (
            scenario_folder / scenario.id / "fleet_opt" / "interim" / role_name
        )
        res_dir_curr.mkdir(parents=True, exist_ok=True)
        clear_folder(res_dir_curr)
        routes_simple, skipped_shipments = fleet_optimization(
            stores_spread,
            month_schedule,
            stores_group,
            role_name,
            res_dir_curr,
            project_params,
        )

        results.append(
            {
                "role_name": role_name,
                "routes_simple": routes_simple,
                "skipped_shipments": skipped_shipments,
            }
        )

    save_routing_results(results, scenario_folder / scenario.id)

    scenario.route_optimization_completed = True
    scenario.save(scenario_folder)

    return "Route Optimization completed successfully!"
