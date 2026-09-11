"""HTML comparison reports."""

from __future__ import annotations

import html
from datetime import datetime, timezone

from .agent_runner import AgentResult
from .baseline import AgentComparison, format_comparison_markdown
from .evaluator import EvaluatorResult
from .git_manager import GitManager, branch_name
from .metrics import format_token_summary


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def generate_html_report(
    run_name: str,
    base_ref: str,
    prompt: str,
    results: list[AgentResult],
    git: GitManager,
    *,
    eval_results: dict[str, list[EvaluatorResult]] | None = None,
    aggregates: dict[str, str] | None = None,
    iteration: int = 1,
    baseline_comparisons: list[AgentComparison] | None = None,
) -> str:
    """Build a self-contained HTML report mirroring the markdown output."""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rows: list[str] = []
    for r in results:
        stats = git.get_diff_stats(r.agent_name, run_name, base_ref)
        status = "pass" if r.exit_code == 0 and not r.error else "fail"
        status_label = "✅" if status == "pass" else "❌"
        if r.error:
            status_label += f" {_esc(r.error)}"
        rows.append(
            "<tr>"
            f"<td>{_esc(r.agent_name)}</td>"
            f'<td class="{status}">{status_label}</td>'
            f"<td>{r.duration:.1f}s</td>"
            f"<td>{_esc(format_token_summary(r.usage))}</td>"
            f"<td>{stats.files_changed}</td>"
            f"<td>+{stats.insertions}</td>"
            f"<td>-{stats.deletions}</td>"
            "</tr>"
        )

    agent_sections: list[str] = []
    for r in results:
        stats = git.get_diff_stats(r.agent_name, run_name, base_ref)
        files = "".join(f"<li><code>{_esc(f)}</code></li>" for f in stats.changed_files)
        branch_ref = _esc(branch_name(r.agent_name, run_name))
        section = [
            f"<h2>{_esc(r.agent_name)}</h2>",
            f"<p><strong>Branch:</strong> <code>{branch_ref}</code></p>",
            f"<p><strong>Duration:</strong> {r.duration:.1f}s</p>",
            f"<p><strong>Exit code:</strong> {r.exit_code}</p>",
        ]
        if r.error:
            section.append(f"<p><strong>Error:</strong> {_esc(r.error)}</p>")
        if files:
            section.append(f"<h3>Files changed</h3><ul>{files}</ul>")
        if eval_results and r.agent_name in eval_results:
            section.append("<h3>Evaluations</h3>")
            for ev in eval_results[r.agent_name]:
                section.append(
                    f"<h4>{_esc(ev.evaluator_name)} ({ev.duration:.1f}s)</h4>"
                    f"<pre>{_esc(ev.critique)}</pre>"
                )
        if aggregates and r.agent_name in aggregates:
            section.append(
                "<h3>Aggregate assessment</h3>"
                f"<pre>{_esc(aggregates[r.agent_name])}</pre>"
            )
        agent_sections.append("\n".join(section))

    baseline_block = ""
    if baseline_comparisons:
        md_lines = format_comparison_markdown(baseline_comparisons)
        baseline_block = (
            "<h2>Baseline comparison</h2><pre>" + _esc("\n".join(md_lines)) + "</pre>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AgentTester Report: {_esc(run_name)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #f4f4f4; }}
    tr.fail td {{ background: #fff0f0; }}
    tr.pass td {{ }}
    pre {{ background: #f8f8f8; padding: 1rem; overflow-x: auto; }}
    code {{ font-family: ui-monospace, monospace; }}
  </style>
</head>
<body>
  <h1>AgentTester Report: {_esc(run_name)}</h1>
  <p><strong>Base ref:</strong> <code>{_esc(base_ref[:12])}</code></p>
  <p><strong>Date:</strong> {now}</p>
  <p><strong>Iteration:</strong> {iteration}</p>
  <p><strong>Agents:</strong> {_esc(", ".join(r.agent_name for r in results))}</p>
  <h2>Prompt</h2>
  <pre>{_esc(prompt)}</pre>
  <h2>Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Agent</th><th>Status</th><th>Duration</th><th>Tokens</th>
        <th>Files</th><th>+</th><th>-</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
  {baseline_block}
  {"".join(agent_sections)}
</body>
</html>
"""
