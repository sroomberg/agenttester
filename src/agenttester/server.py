"""HTTP receiver for agent completion callbacks."""

from __future__ import annotations

import asyncio

from aiohttp import web
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


def _make_app(console: Console) -> web.Application:
    async def handle_result(request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
        model = str(data.get("model", "unknown"))
        result = str(data.get("result", ""))
        branch = data.get("branch")
        subtitle = f"[dim]{branch}[/dim]" if branch else None
        console.print(
            Panel(
                Markdown(result),
                title=f"[bold]{model}[/bold]",
                subtitle=subtitle,
                border_style="green",
            )
        )
        console.print()
        return web.json_response({"ok": True})

    async def handle_health(request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    app = web.Application()
    app.router.add_post("/result", handle_result)
    app.router.add_get("/health", handle_health)
    return app


async def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    console = Console()
    app = _make_app(console)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    url = f"http://{host}:{port}"
    console.print(f"[bold]agent-tester receiver[/bold]  {url}")
    console.print(f'[dim]POST {url}/result  {{"model": "...", "result": "..."}}[/dim]')
    console.print("[dim]Ctrl-C to stop[/dim]\n")

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
