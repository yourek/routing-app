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

    # optional / default fields last
    role_spec_filled: dict
    visits_window: Tuple[int, int] = field(default_factory=lambda: (0, 24))
    number_of_ftes: float = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    # relative hours from Delivery Day

    def to_dict(self) -> dict:
        """Convert role to dict (e.g., for JSON or saving to DB)."""
        return {
            "project_id": self.project_id,
            "scenario_id": self.scenario_id,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "role_spec_filled": self.role_spec_filled,
            "visits_window": self.visits_window,
            "number_of_ftes": self.number_of_ftes,
        }

    def __str__(self) -> str:
        return (
            f"Role(name={self.name}, "
            f"window={self.visits_window[0]}h→{self.visits_window[1]}h)"
        )
