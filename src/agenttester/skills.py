"""Skills loader — discovers and merges per-session prompt instructions."""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import GLOBAL_CONFIG_DIR, get_config_paths

_BUILTIN_SKILLS_DIR = Path(__file__).parent / "skills"


def _get_global_skills_dir() -> Path | None:
    """Return the first existing global skills directory."""
    candidates = [
        GLOBAL_CONFIG_DIR / "skills",
        Path.home() / ".agenttester" / "skills",
    ]
    return next((p for p in candidates if p.is_dir()), None)


def _load_dir(directory: Path) -> dict[str, str]:
    """Return {filename: content} for all .md files in *directory*."""
    return {p.name: p.read_text() for p in sorted(directory.glob("*.md"))}


def _load_extra(paths: list[Path]) -> list[str]:
    """Load skills from a list of file or directory paths, in order."""
    sections: list[str] = []
    for p in paths:
        if p.is_dir():
            for f in sorted(p.glob("*.md")):
                text = f.read_text().strip()
                if text:
                    sections.append(text)
        elif p.is_file() and p.suffix == ".md":
            text = p.read_text().strip()
            if text:
                sections.append(text)
    return sections


def _skills_from_configs(repo_path: Path | None) -> list[Path]:
    """Read the ``skills:`` key from all config files and return resolved paths."""
    try:
        result: list[Path] = []
        for cfg_path in get_config_paths(repo_path):
            with open(cfg_path) as f:
                data = yaml.safe_load(f) or {}
            for entry in data.get("skills") or []:
                p = Path(entry).expanduser()
                if not p.is_absolute():
                    p = cfg_path.parent / p
                result.append(p)
        return result
    except Exception:
        return []


def load_skills(
    repo_path: Path | None = None,
    extra_paths: list[Path] | None = None,
) -> str:
    """Return combined skill instructions to prepend to every agent prompt.

    Skills are output in priority order so that higher-priority instructions
    appear later in the prompt (recency bias):
      built-ins → global user skills → local project skills
      → config skills → extra_paths

    A user skill with the same filename as a built-in replaces it entirely and
    still appears at the end, ensuring user intent always takes precedence.
    """
    builtin = _load_dir(_BUILTIN_SKILLS_DIR)

    global_dir = _get_global_skills_dir()
    global_skills = _load_dir(global_dir) if global_dir else {}

    local_skills: dict[str, str] = {}
    if repo_path:
        local_dir = repo_path / ".agent-tester" / "skills"
        if local_dir.is_dir():
            local_skills = _load_dir(local_dir)

    overridden_by_local = set(local_skills)
    overridden_by_any = set(global_skills) | overridden_by_local

    sections: list[str] = []
    for name, content in builtin.items():
        if name not in overridden_by_any and content.strip():
            sections.append(content.strip())
    for name, content in global_skills.items():
        if name not in overridden_by_local and content.strip():
            sections.append(content.strip())
    for content in local_skills.values():
        if content.strip():
            sections.append(content.strip())

    # Skills declared in config files (global then local)
    sections.extend(_load_extra(_skills_from_configs(repo_path)))

    # Skills passed explicitly at runtime (highest priority)
    if extra_paths:
        sections.extend(_load_extra(extra_paths))

    return "\n\n".join(sections)
