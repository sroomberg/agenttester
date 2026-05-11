"""Skills loader — discovers and merges per-session prompt instructions."""

from __future__ import annotations

from pathlib import Path

_BUILTIN_SKILLS_DIR = Path(__file__).parent / "skills"


def _get_global_skills_dir() -> Path | None:
    """Return the first existing global skills directory."""
    home = Path.home()
    candidates = [
        home / ".config" / "agenttester" / "skills",
        home / ".agenttester" / "skills",
    ]
    return next((p for p in candidates if p.is_dir()), None)


def _load_dir(directory: Path) -> dict[str, str]:
    """Return {filename: content} for all .md files in *directory*."""
    return {p.name: p.read_text() for p in sorted(directory.glob("*.md"))}


def load_skills(repo_path: Path | None = None) -> str:
    """Return combined skill instructions to prepend to every agent prompt.

    Merge order (highest priority wins by filename):
      local (.agent-tester/skills/) > global (~/.agenttester/skills/) > built-ins
    """
    merged: dict[str, str] = _load_dir(_BUILTIN_SKILLS_DIR)

    global_dir = _get_global_skills_dir()
    if global_dir:
        merged.update(_load_dir(global_dir))

    if repo_path:
        local_dir = repo_path / ".agent-tester" / "skills"
        if local_dir.is_dir():
            merged.update(_load_dir(local_dir))

    if not merged:
        return ""

    sections = [content.strip() for content in merged.values() if content.strip()]
    return "\n\n".join(sections)
