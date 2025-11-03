import numpy as np
import os
import shutil
import calendar
from datetime import datetime

from params.regionalization import (
    BASE_DATE,
    SECONDS_IN_HOUR,
    WEEKS_IN_MONTH,
)


def expanded_data(X, sample_weights, hours_accuracy):
    indices = []
    for i, w in enumerate(sample_weights):
        n_copies = int(round(w * hours_accuracy))
        indices.extend([i] * max(n_copies, 1))
    return X.loc[indices], np.array(indices)


def clear_folder(folder_path: str):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove directory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def map_workweek_days(key):
    mapping = {
        "Sunday - Thursday": [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
        ],
        "Monday - Friday": [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
        ],
    }

    return mapping[key]


def get_dayname_to_daynum_mapping():
    day_to_num = {day_: (i + 1) % 7 for i, day_ in enumerate(calendar.day_name)}

    return day_to_num


def get_max_month_store_hours(project_params, scenario):
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
    max_month_store_hours = int(
        round(max_month_hours * scenario.time_in_store_used_in_rgn / 100)
    )

    return max_month_store_hours
