import json
from pathlib import Path


def load_default_input_parameters(project_id: str) -> dict:
    project_data_path = Path("data") / "projects" / project_id
    project_data_path.mkdir(parents=True, exist_ok=True)
    params_path = project_data_path / "input_parameters.json"
    with open(params_path, "r", encoding="utf-8") as f:
        input_parameter_dict = json.load(f).copy()

    return input_parameter_dict
