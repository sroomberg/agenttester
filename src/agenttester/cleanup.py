"""Interactive branch cleanup for agenttester sessions."""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit.shortcuts import (
    checkboxlist_dialog,
    radiolist_dialog,
    yes_no_dialog,
)
from rich.console import Console

from .git_manager import GitManager
from .session import ReplSession


def run_cleanup(workdir: Path, remote: str = "origin") -> None:
    console = Console()

    try:
        git_mgr = GitManager(workdir)
    except Exception:
        console.print("[red]Not a git repository.[/red]")
        return

    all_sessions = ReplSession.list_all()
    sessions_with_branches = [s for s in all_sessions if s.branches]

    if not sessions_with_branches:
        console.print("[dim]No sessions with recorded branches found.[/dim]")
        return

    local_branches = set(git_mgr.list_agenttester_branches())
    remote_branches = set(git_mgr.list_remote_agenttester_branches(remote))
    all_branches = local_branches | remote_branches

    def _existing(s: ReplSession) -> list[str]:
        return [b for b in s.branches if b in all_branches]

    sessions_to_show = [s for s in sessions_with_branches if _existing(s)]

    if not sessions_to_show:
        console.print(
            "[dim]No sessions with existing local or remote branches found.[/dim]"
        )
        return

    n = len(sessions_to_show)
    console.print(f"[dim]Found {n} session(s) with existing branches.[/dim]\n")

    # ── Phase 1: select sessions to delete entirely ───────────────────────────
    full_delete_ids: list[str] | None = checkboxlist_dialog(
        title="Step 1 — Delete sessions entirely",
        text=(
            "Select sessions whose branches you want to DELETE ENTIRELY.\n"
            "Leave all unchecked to move on to individual branch selection."
        ),
        values=[
            (
                s.id,
                f"{s.id}  ({len(_existing(s))} branch(es))",
            )
            for s in sessions_to_show
        ],
    ).run()

    if full_delete_ids is None:
        console.print("[dim]Cancelled.[/dim]")
        return

    remaining_sessions = [s for s in sessions_to_show if s.id not in full_delete_ids]

    # ── Phase 2: individual branch selection for remaining sessions ───────────
    individual_branches: list[str] = []

    for s in remaining_sessions:
        existing = _existing(s)
        if not existing:
            continue

        selected: list[str] | None = checkboxlist_dialog(
            title=f"Step 2 — Session: {s.id}",
            text=(
                "Select individual branches to delete.\n"
                "Leave all unchecked to keep everything for this session."
            ),
            values=[(b, b) for b in existing],
        ).run()

        if selected is None:
            console.print("[dim]Cancelled.[/dim]")
            return

        individual_branches.extend(selected)

    # ── Collect all branches to delete ────────────────────────────────────────
    full_delete_branches = [
        b for s in sessions_to_show if s.id in full_delete_ids for b in _existing(s)
    ]
    all_to_delete = list(dict.fromkeys(full_delete_branches + individual_branches))

    if not all_to_delete:
        console.print("[dim]Nothing selected for deletion.[/dim]")
        return

    # ── Ask deletion scope ────────────────────────────────────────────────────
    mode: str | None = radiolist_dialog(
        title="Deletion scope",
        text=f"How should the {len(all_to_delete)} selected branch(es) be deleted?",
        values=[
            ("both", "Local and remote"),
            ("local", "Local only"),
            ("remote", "Remote only"),
        ],
        default="both",
    ).run()

    if mode is None:
        console.print("[dim]Cancelled.[/dim]")
        return

    # ── Confirm ───────────────────────────────────────────────────────────────
    scope_label = {
        "both": "locally and remotely",
        "local": "locally only",
        "remote": "from remote only",
    }[mode]

    confirmed: bool = yes_no_dialog(
        title="Confirm deletion",
        text=f"Delete {len(all_to_delete)} branch(es) {scope_label}?",
    ).run()

    if not confirmed:
        console.print("[dim]Cancelled.[/dim]")
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    console.print()
    for branch in all_to_delete:
        if mode in ("local", "both"):
            ok = git_mgr.delete_local_branch(branch)
            icon = "[green]✓[/green]" if ok else "[yellow]~[/yellow]"
            console.print(f"  {icon} local   {branch}")
        if mode in ("remote", "both"):
            ok = git_mgr.delete_remote_branch(branch, remote)
            icon = "[green]✓[/green]" if ok else "[yellow]~[/yellow]"
            console.print(f"  {icon} remote  {branch}")

    # ── Optionally delete session records for fully-deleted sessions ──────────
    fully_deleted_sessions = [
        s for s in sessions_to_show if s.id in full_delete_ids
    ]
    if fully_deleted_sessions:
        console.print()
        delete_records: bool = yes_no_dialog(
            title="Delete session records?",
            text=(
                f"Also delete the {len(fully_deleted_sessions)} session record(s)"
                " (conversation history, reports, eval results)?\n\n"
                "Choose 'No' to keep them for reference."
            ),
        ).run()
        if delete_records:
            for s in fully_deleted_sessions:
                s.delete()
                console.print(f"  [green]✓[/green] session  {s.id}")

    console.print("\n[green]Done.[/green]")
