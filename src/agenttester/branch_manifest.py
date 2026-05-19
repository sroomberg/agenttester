"""Persistent manifest of agenttester branches and their remote origins."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import GLOBAL_CONFIG_DIR


def _manifest_path() -> Path:
    return GLOBAL_CONFIG_DIR / "branches.json"


def _load(path: Path | None = None) -> dict:
    p = path or _manifest_path()
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data: dict, path: Path | None = None) -> None:
    p = path or _manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def record_branch(
    branch: str,
    remote: str,
    session_id: str,
    path: Path | None = None,
) -> None:
    """Add or update a branch entry in the manifest."""
    data = _load(path)
    data[branch] = {
        "remote": remote,
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save(data, path)


def remove_branch(branch: str, path: Path | None = None) -> None:
    """Remove a branch from the manifest."""
    data = _load(path)
    if branch in data:
        data.pop(branch)
        _save(data, path)


def list_branches(path: Path | None = None) -> dict[str, dict]:
    """Return all manifest entries keyed by branch name."""
    return _load(path)
