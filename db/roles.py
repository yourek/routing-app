from dataclasses import dataclass, field
from typing import Tuple
import uuid


@dataclass
class Role:
    # required fields first
    project_id: str
    scenario_id: str
    name: str
    description: str
    default_visits: int
    default_hours: float

    # optional / default fields last
    visits_window: Tuple[int, int] = field(default_factory=lambda: (0, 24))
    number_of_ftes: float = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # relative hours from Delivery Day

    def to_dict(self) -> dict:
        """Convert role to dict (e.g., for JSON or saving to DB)."""
        return {
            "project_id": self.project_id,
            "scenario_id": self.scenario_id,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "default_visits": self.default_visits,
            "default_hours": self.default_hours,
            "visits_window": self.visits_window,
            "number_of_ftes": self.number_of_ftes,
        }

    def __str__(self) -> str:
        return (
            f"Role(name={self.name}, visits={self.default_visits}/month, "
            f"hours={self.default_hours}/month, "
            f"window={self.visits_window[0]}h→{self.visits_window[1]}h)"
        )
