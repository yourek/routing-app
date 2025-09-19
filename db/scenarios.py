from dataclasses import dataclass, field, asdict
from typing import List, Optional
from datetime import datetime
import uuid
import json
from pathlib import Path

from db.roles import Role


@dataclass
class Scenario:
    project_id: str
    name: str
    description: str = ""
    author: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    roles: List[Role] = field(default_factory=list)

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
        file_path = folder_path / f"{self.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

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
                json_path = subfolder / f"{subfolder.name}.json"
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
