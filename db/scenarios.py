import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from datetime import datetime
import uuid
import json
from pathlib import Path
import shutil
import pandas as pd
import json

from db.roles import Role


@dataclass
class Scenario:
    project_id: str
    name: str
    description: str = ""
    author: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    roles: List[Role] = field(default_factory=list)
    time_in_store: int = field(default_factory=lambda: 83)  # Default to 83%
    time_in_store_used_in_rgn: int = field(default_factory=lambda: 83)

    # Boolean flags for process completion
    regionalization_completed: bool = False
    week_distribution_completed: bool = False
    route_optimization_completed: bool = False

    # ------------------------
    # Methods to mark completion
    # ------------------------
    def complete_regionalization(self):
        # Run your process here
        self.regionalization_completed = True
        self.modified_at = datetime.now()

    def complete_week_distribution(self):
        # Run your process here
        self.week_distribution_completed = True
        self.modified_at = datetime.now()

    def complete_route_optimization(self):
        # Run your process here
        self.route_optimization_completed = True
        self.modified_at = datetime.now()

    # Add role to scenario
    def add_role(self, role: Role):
        self.roles.append(role)
        self.modified_at = datetime.now()

    def update_stores_w_role_spec(self, folder: Path, role: Role):
        input_output_path = folder / self.id / "stores_w_role_spec.json"
        with open(input_output_path, "r", encoding="utf-8") as f:
            stores_w_role_spec = pd.DataFrame(json.load(f))
        role_spec_filled_df = pd.DataFrame(role.role_spec_filled)
        role_spec_filled_df[["Group", "Grading"]] = role_spec_filled_df[
            ["Group", "Grading"]
        ].astype(str)
        stores_w_role_spec = pd.merge(
            stores_w_role_spec,
            role_spec_filled_df,
            how="left",
            on=["Group", "Grading"],
            validate="m:1",
        )
        with open(input_output_path, "w", encoding="utf-8") as f:
            json.dump(
                stores_w_role_spec.to_dict(orient="records"),
                f,
                ensure_ascii=False,
                indent=2,
            )

    def estimate_route_optimization(self, folder: Path):
        stores_w_role_spec_path = folder / self.id / "stores_w_role_spec.json"
        with open(stores_w_role_spec_path, "r", encoding="utf-8") as f:
            stores_w_role_spec = pd.DataFrame(json.load(f))

        frequency_cols = [
            c for c in stores_w_role_spec.columns if " - visits per month" in c
        ]
        if frequency_cols:
            frequency_total = stores_w_role_spec[frequency_cols].sum().sum()
            cost_total = frequency_total * 30 / 1_000
            return cost_total

    # Optional: summary method
    # ------------------------
    # JSON persistence
    # ------------------------
    def to_dict(self):
        data = asdict(self)
        # Convert roles to dicts
        data["roles"] = [asdict(r) for r in self.roles]
        data["created_at"] = str(data["created_at"])
        data["modified_at"] = str(data["modified_at"])

        return data

    def save(self, folder: Path):
        folder_path = folder / self.id
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path = folder_path / f"scenario_metadata.json"
        self.modified_at = datetime.now()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def save_initial_stores_w_role_spec(self, folder: Path):
        input_path = folder.parent.parent / self.project_id / "stores.json"
        output_folder_path = folder / self.id
        output_folder_path.mkdir(parents=True, exist_ok=True)
        output_path = output_folder_path / "stores_w_role_spec.json"
        _ = shutil.copy2(input_path, output_path)

    def remove_cols_from_stores_w_role_spec(self, folder: Path, role_name: str):
        folder_path = folder / self.id
        file_path = folder_path / "stores_w_role_spec.json"
        with open(file_path, "r", encoding="utf-8") as f:
            stores_w_role_spec = pd.DataFrame(json.load(f))
        stores_w_role_spec = stores_w_role_spec.drop(
            columns=[
                f"{role_name} - visits per month",
                f"{role_name} - time in store",
            ]
        )
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(stores_w_role_spec.to_dict(), f, ensure_ascii=False, indent=2)

    def remove_role_results(self, folder: Path, role_name: str):
        folder_path = folder / self.id
        files_to_potentially_delete = [
            folder_path / "rgn_res" / f"{role_name}.json",
            folder_path / "week_dist_res" / f"{role_name} - month_schedule.json",
            folder_path / "week_dist_res" / f"{role_name} - stores_spread.json",
            folder_path / "fleet_opt" / "final" / f"{role_name} - routes_simple.json",
            folder_path
            / "fleet_opt"
            / "final"
            / f"{role_name} - skipped_shipments.json",
        ]
        for file_path in files_to_potentially_delete:
            if os.path.exists(file_path):
                os.remove(file_path)

    def update_process_step_status(self, folder):
        folder_path = folder / self.id
        rgn_res_folder_path = folder_path / "rgn_res"
        if not os.path.exists(rgn_res_folder_path) or not os.listdir(
            rgn_res_folder_path
        ):
            self.regionalization_completed = False
        week_dist_res_folder_path = folder_path / "week_dist_res"
        if not os.path.exists(week_dist_res_folder_path) or not os.listdir(
            week_dist_res_folder_path
        ):
            self.week_distribution_completed = False
        fleet_opt_folder_path = folder_path / "fleet_opt" / "final"
        if not os.path.exists(fleet_opt_folder_path) or not os.listdir(
            fleet_opt_folder_path
        ):
            self.route_optimization_completed = False

    @staticmethod
    def load(file_path: Path) -> "Scenario":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        roles = [Role(**r) for r in data.get("roles", [])]

        # cast timestamps
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()

        modified_at = data.get("modified_at")
        if isinstance(modified_at, str):
            modified_at = datetime.fromisoformat(modified_at)
        else:
            modified_at = datetime.now()

        return Scenario(
            project_id=data["project_id"],
            name=data["name"],
            description=data.get("description", ""),
            author=data.get("author", ""),
            id=data.get("id", str(uuid.uuid4())),
            created_at=created_at,
            modified_at=modified_at,
            time_in_store=data.get("time_in_store", 100),
            time_in_store_used_in_rgn=data.get("time_in_store_used_in_rgn", 100),
            roles=roles,
            regionalization_completed=data.get("regionalization_completed", False),
            week_distribution_completed=data.get("week_distribution_completed", False),
            route_optimization_completed=data.get(
                "route_optimization_completed", False
            ),
        )

    @staticmethod
    def load_all(folder: Path) -> List["Scenario"]:
        folder.mkdir(parents=True, exist_ok=True)
        scenarios = []

        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                json_path = subfolder / f"scenario_metadata.json"
                if json_path.exists():
                    try:
                        scenarios.append(Scenario.load(json_path))
                    except Exception as e:
                        print(f"⚠️ Failed to load {json_path}: {e}")
                else:
                    print(f"⚠️ No JSON file found in {subfolder}")

        # for file_path in folder.glob("*.json"):
        #     try:
        #         scenarios.append(Scenario.load(file_path))
        #     except Exception as e:
        #         print(f"Failed to load {file_path}: {e}")

        return scenarios


def load_scenarios() -> list[dict]:
    pass
