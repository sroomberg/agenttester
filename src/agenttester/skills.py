"""Skills loader — discovers and merges per-session prompt instructions."""

from __future__ import annotations

from pathlib import Path

from .config import GLOBAL_CONFIG_DIR

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


def load_skills(repo_path: Path | None = None) -> str:
    """Return combined skill instructions to prepend to every agent prompt.

    Skills are output in priority order so that higher-priority instructions
    appear later in the prompt (recency bias):
      built-ins → global user skills → local project skills

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

    return "\n\n".join(sections)
