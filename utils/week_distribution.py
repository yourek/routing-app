import numpy as np
import pandas as pd
import json
from scipy.stats import mode
from k_means_constrained import KMeansConstrained
from datetime import timedelta, datetime

from utils.models_common import (
    expanded_data,
    map_workweek_days,
    get_dayname_to_daynum_mapping,
)
from params.regionalization import (
    WEEKS_IN_MONTH,
    MONTH_TO_FOUR_WEEKS_ADJUSTER,
    BASE_DATE,
)


def get_day_to_next_day_mapping(workweek_days):
    day_to_next_day = {
        workweek_days[i]: workweek_days[(i + 1) % len(workweek_days)]
        for i in range(len(workweek_days))
    }
    day_to_prv_day = {v: k for k, v in day_to_next_day.items()}

    return day_to_next_day, day_to_prv_day


def apply_mapping_n_times(value, forward_map, backward_map, n):
    """
    Repeatedly applies forward_map or backward_map to a value.

    Args:
        value: starting value
        forward_map (dict): mapping used if n > 0
        backward_map (dict): mapping used if n < 0
        n (int): number of steps (positive = forward, negative = backward)

    Returns:
        The transformed value after applying the mapping.
    """
    if n == 0:
        return value

    mapping = forward_map if n > 0 else backward_map
    steps = abs(n)

    for _ in range(steps):
        value = mapping[value]

    return value


def handle_biweekly_visit(stores, freq_column, hours_column, sales_rep):
    # Preparing biweekly visit cases
    rep_stores_biweekly = (
        stores.loc[stores[freq_column].eq(2) & stores["Sales Rep name"].eq(sales_rep)]
        .reset_index(drop=True)
        .copy()
    )

    X = rep_stores_biweekly[["Latitude", "Longitude"]]
    X_expanded, mapping = expanded_data(
        X, rep_stores_biweekly[hours_column], hours_accuracy=1
    )
    size_max = np.ceil(X_expanded.shape[0] / 2)

    if rep_stores_biweekly.shape[0] >= 2:
        kmeans = KMeansConstrained(n_clusters=2, size_max=size_max, random_state=2025)
        labels_expanded = kmeans.fit_predict(X_expanded)

        clusters = pd.DataFrame({"index": mapping, "Cluster": labels_expanded})
        clusters = clusters.groupby("index").agg({"Cluster": mode})
        clusters["Cluster"] = (clusters["Cluster"].str[0].astype("int")) + 1
    else:
        clusters = pd.DataFrame({"index": rep_stores_biweekly.index}).set_index("index")
        clusters["Cluster"] = 1

    rep_stores_biweekly = rep_stores_biweekly.merge(
        clusters, how="left", left_index=True, right_index=True
    )

    rep_stores_biweekly = pd.concat(
        [
            rep_stores_biweekly.assign(
                Week=lambda df: "Week" + (df["Cluster"] + (i * 2)).astype(str),
                VisitDays="0,1,2,3,4",
            )
            for i in range(2)
        ]
    )
    return rep_stores_biweekly


def handle_monthly_visits(stores, freq_column, hours_column, sales_rep):
    # Preparing monthly visit cases
    rep_stores_monthly = (
        stores.loc[stores[freq_column].eq(1) & stores["Sales Rep name"].eq(sales_rep)]
        .reset_index(drop=True)
        .copy()
    )

    X = rep_stores_monthly[["Latitude", "Longitude"]]
    X_expanded, mapping = expanded_data(
        X, rep_stores_monthly[hours_column], hours_accuracy=1
    )
    size_max = np.ceil(X_expanded.shape[0] / 4)

    if rep_stores_monthly.shape[0] >= 4:
        kmeans = KMeansConstrained(n_clusters=4, size_max=size_max, random_state=2025)
        labels_expanded = kmeans.fit_predict(X_expanded)

        clusters = pd.DataFrame({"index": mapping, "Cluster": labels_expanded})
        clusters = clusters.groupby("index").agg({"Cluster": mode})
        clusters["Cluster"] = (clusters["Cluster"].str[0].astype("int")) + 1
    else:
        clusters = pd.DataFrame({"index": rep_stores_monthly.index}).set_index("index")
        clusters["Cluster"] = 1

    rep_stores_monthly = rep_stores_monthly.merge(
        clusters, how="left", left_index=True, right_index=True
    )

    rep_stores_monthly["Week"] = "Week" + rep_stores_monthly["Cluster"].astype(str)
    rep_stores_monthly["VisitDays"] = "0,1,2,3,4"

    return rep_stores_monthly


def cluster_multivisit_stores(stores, freq_column, hours_column, traveling_sales_reps):
    stores_spread = list()
    for sales_rep in traveling_sales_reps:

        # Preparing 5 weekly visit cases
        rep_stores_5weekly = stores.loc[
            stores[freq_column].eq(20) & stores["Sales Rep name"].eq(sales_rep)
        ].copy()
        rep_stores_5weekly = pd.concat(
            [
                rep_stores_5weekly.assign(
                    Cluster=int(i / 5),
                    Week="Week" + str(int((i / 5) + 1)),
                    VisitDays=str(i % 5),
                )
                for i in range(20)
            ]
        )

        # Preparing 4 weekly visit cases
        rep_stores_4weekly = stores.loc[
            stores[freq_column].eq(16) & stores["Sales Rep name"].eq(sales_rep)
        ].copy()
        rep_stores_4weekly = pd.concat(
            [
                rep_stores_4weekly.assign(
                    Cluster=int(i / 4),
                    Week="Week" + str(int((i / 4) + 1)),
                    VisitDays=(
                        "0,1"
                        if i % 4 == 0
                        else ("2" if i % 4 == 1 else ("3" if i % 4 == 2 else "4"))
                    ),
                )
                for i in range(16)
            ]
        )

        # Preparing 3 weekly visit cases
        rep_stores_3weekly = stores.loc[
            stores[freq_column].eq(12) & stores["Sales Rep name"].eq(sales_rep)
        ].copy()
        rep_stores_3weekly = pd.concat(
            [
                rep_stores_3weekly.assign(
                    Cluster=int(i / 3),
                    Week="Week" + str(int((i / 3) + 1)),
                    VisitDays="0,1" if i % 3 == 0 else ("2,3" if i % 3 == 1 else "4"),
                )
                for i in range(12)
            ]
        )

        # Preparing 2 weekly visit cases
        rep_stores_2weekly = stores.loc[
            stores[freq_column].eq(8) & stores["Sales Rep name"].eq(sales_rep)
        ].copy()
        rep_stores_2weekly = pd.concat(
            [
                rep_stores_2weekly.assign(
                    Cluster=int(i / 2),
                    Week="Week" + str(int((i / 2) + 1)),
                    VisitDays="0,1,2" if i % 2 == 0 else "3,4",
                )
                for i in range(8)
            ]
        )

        # Preparing weekly visit cases
        rep_stores_weekly = stores.loc[
            stores[freq_column].eq(4) & stores["Sales Rep name"].eq(sales_rep)
        ].copy()
        rep_stores_weekly = pd.concat(
            [
                rep_stores_weekly.assign(
                    Cluster=i, Week="Week" + str(i + 1), VisitDays="0,1,2,3,4"
                )
                for i in range(4)
            ]
        )

        # Preparing clusters biweekly visit cases
        rep_stores_biweekly = handle_biweekly_visit(
            stores, freq_column, hours_column, sales_rep
        )

        # Preparing clusters monthly visit cases
        rep_stores_monthly = handle_monthly_visits(
            stores, freq_column, hours_column, sales_rep
        )

        # Complete case
        rep_stores_spread = pd.concat(
            [
                rep_stores_5weekly,
                rep_stores_4weekly,
                rep_stores_3weekly,
                rep_stores_2weekly,
                rep_stores_weekly,
                rep_stores_biweekly,
                rep_stores_monthly,
            ]
        )
        stores_spread.append(rep_stores_spread)

    # Collecting clustered results
    stores_spread = pd.concat(stores_spread, ignore_index=True)

    return stores_spread


def add_distribution_days_from_day_columns(stores, workweek_days):
    stores["Distribution Days"] = stores.apply(
        lambda r: " , ".join(
            d for d in workweek_days if (not pd.isna(r[d])) and (r[d])
        ),
        axis=1,
    )
    stores["Distribution Days"] = stores["Distribution Days"].replace("", "0")
    return stores


def prepare_visit_days(
    stores_spread,
    freq_column,
    workweek_days,
    visits_window_in_days,
):
    visit_days_senior = stores_spread[
        ["Customer", "Chain", "Distribution Days", freq_column, "VisitDays"]
    ].copy()

    day_to_next_day, day_to_prv_day = get_day_to_next_day_mapping(workweek_days)

    visit_days_senior[workweek_days] = False
    for day in workweek_days:
        window_days = [
            apply_mapping_n_times(day, day_to_next_day, day_to_prv_day, day_int)
            for day_int in visits_window_in_days
        ]
        for window_day in window_days:
            visit_days_senior[window_day] = (
                visit_days_senior[window_day] | stores_spread[day]
            )

    visit_days_senior.loc[
        (
            visit_days_senior["Distribution Days"].isin(["0", "Everyday"])
            | visit_days_senior["Chain"].isin(["Be Pharm", "Shufersal"])
        ),
        workweek_days,
    ] = True

    visit_days_senior["Days"] = (
        visit_days_senior[workweek_days] * ["0", "1", "2", "3", "4"]
    ).apply(lambda row: [d for d in row if d != ""], axis=1)
    visit_days_senior.loc[visit_days_senior[freq_column].eq(8), "Days"] = (
        visit_days_senior.apply(
            lambda r: (
                r["Days"][:2]
                if r["VisitDays"] == ("0,1,2")
                else (r["Days"][2:] if r["VisitDays"] == ("3,4") else r["Days"])
            ),
            axis=1,
        )
    )
    visit_days_senior["VisitDays"] = (
        visit_days_senior["Days"].apply(lambda x: sorted(set(x))).str.join(",")
    )

    return visit_days_senior


def save_week_dist_results(results, scenario_path):
    week_dist_res_folder = scenario_path / "week_dist_res"
    week_dist_res_folder.mkdir(parents=True, exist_ok=True)

    for result in results:
        role_name = result["role_name"]
        with open(
            week_dist_res_folder / f"{role_name} - stores_spread.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result["stores_spread"].to_dict(orient="records"),
                f,
                ensure_ascii=False,
                indent=2,
            )
        with open(
            week_dist_res_folder / f"{role_name} - month_schedule.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result["month_schedule"].to_dict(orient="records"),
                f,
                ensure_ascii=False,
                indent=2,
            )


def week_distribution(scenario_folder, scenario):
    scenario_json_path = scenario_folder / scenario.id / "scenario_metadata.json"
    stores_json_path = scenario_folder / scenario.id / "stores_w_role_spec.json"
    project_params_path = scenario_folder.parent / "input_parameters.json"

    with open(scenario_json_path, "r") as f:
        scen_metadata = json.load(f)
    with open(stores_json_path, "r", encoding="utf-8") as f:
        stores = pd.DataFrame(json.load(f))
    with open(project_params_path, "r", encoding="utf-8") as f:
        project_params = json.load(f)

    workweek_days = map_workweek_days(project_params["week_schedule"])
    days_in_workweek = len(workweek_days)
    day_to_num = get_dayname_to_daynum_mapping()

    results = list()
    for role in scen_metadata["roles"]:
        role_name = role["name"]
        visits_window = role["visits_window"]
        visits_window_in_days = list(
            np.arange(visits_window[0] / 24, visits_window[-1] / 24 + 1, 1, int)
        )
        try:
            with open(
                scenario_folder / scenario.id / "rgn_res" / f"{role_name}.json",
                "r",
                encoding="utf-8",
            ) as f:
                stores_group = pd.DataFrame(json.load(f))
        except FileNotFoundError:
            continue

        sales_reps = sorted(set(stores_group["Sales Rep name"]))
        stores_spread = cluster_multivisit_stores(
            stores=stores_group,
            freq_column=f"{role_name} - visits per month",
            hours_column=f"{role_name} - time in store",
            traveling_sales_reps=sales_reps,
        )
        stores_spread = add_distribution_days_from_day_columns(
            stores_spread, workweek_days
        )

        # 2.2 Check if anything has been lost
        print(
            "Working on...\n",
            role_name,
            "\nstores_group hours:",
            stores_group[f"{role_name} - visits per month"].sum(),
            "\nstores_spread hours:",
            stores_spread["Visit Duration"].sum(),
            "\nstores raw hours:",
            (
                stores[f"{role_name} - time in store"] * MONTH_TO_FOUR_WEEKS_ADJUSTER
            ).sum(),
        )

        # 3. Adding distribution day constraint if selected
        if project_params["consider_delivery"]:
            visit_days_senior = prepare_visit_days(
                stores_spread,
                f"{role_name} - visits per month",
                workweek_days,
                visits_window_in_days,
            )
            stores_spread = stores_spread.drop(columns="VisitDays").merge(
                visit_days_senior[["VisitDays"]],
                how="left",
                left_index=True,
                right_index=True,
            )

        # 4. Preparing month schedule for vehicles/sales reps
        workday_start = datetime.strptime(project_params["work_start"], "%H:%M").time()
        workday_end = datetime.strptime(project_params["work_end"], "%H:%M").time()

        month_schedule = list()
        for sales_rep_type in sales_reps:
            for wi in range(WEEKS_IN_MONTH):
                for day in workweek_days:
                    di = day_to_num[day]
                    workday = BASE_DATE + timedelta(weeks=wi, days=di)
                    day_name = workday.strftime("%A")

                    start = workday.replace(
                        hour=workday_start.hour, minute=workday_start.minute
                    )
                    end = workday.replace(
                        hour=workday_end.hour,
                        minute=workday_end.minute,
                    )
                    soft_end = workday.replace(
                        hour=workday_end.hour, minute=workday_end.minute
                    )
                    break_time_earliest = workday.replace(hour=12, minute=0)
                    break_time_latest = workday.replace(
                        hour=workday_end.hour - 4, minute=workday_end.minute
                    )
                    month_schedule.append(
                        [
                            sales_rep_type,
                            day_name,
                            f"Week{wi+1}",
                            start.isoformat() + "Z",
                            end.isoformat() + "Z",
                            soft_end.isoformat() + "Z",
                            break_time_earliest.isoformat() + "Z",
                            break_time_latest.isoformat() + "Z",
                        ]
                    )

        month_schedule = pd.DataFrame(
            month_schedule,
            columns=[
                "sales_rep",
                "day_name",
                "week",
                "start",
                "end",
                "soft_end",
                "break_time_earliest",
                "break_time_latest",
            ],
        )
        results.append(
            {
                "role_name": role_name,
                "stores_spread": stores_spread,
                "month_schedule": month_schedule,
            }
        )
    save_week_dist_results(results, scenario_folder / scenario.id)

    scenario.week_distribution_completed = True
    scenario.save(scenario_folder)

    return "Week Distribution completed successfully!"
