import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from session.session import get_active_project, init_session, set_active_project
import streamlit as st


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def _proj_path(pid: str) -> Path:
    return DATA_DIR / f"{pid}.json"


def load_projects():
    projs = []
    for p in sorted(DATA_DIR.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
                projs.append(d)
        except Exception:
            # Skip unreadable files
            continue

    # Sort: active first, then newest
    projs.sort(key=lambda d: (not d.get("active", False), d.get("created_at", "")), reverse=True)
    return projs


def save_project(d: dict):
    pid = d["id"]
    with open(_proj_path(pid), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def delete_project(pid: str):
    try:
        _proj_path(pid).unlink(missing_ok=True)
        active_project = get_active_project()
        if active_project:
            if pid == active_project["id"]:
                set_active_project(None)

    except Exception:
        pass
