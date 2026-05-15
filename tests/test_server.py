"""Tests for agenttester.server."""

from __future__ import annotations

from io import StringIO

from aiohttp.test_utils import TestClient, TestServer
from rich.console import Console

from agenttester.server import _make_app


def _quiet_console() -> Console:
    return Console(file=StringIO())


class TestHealthEndpoint:
    async def test_returns_ok(self) -> None:
        app = _make_app(_quiet_console())
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"


class TestResultEndpoint:
    async def test_accepts_valid_payload(self) -> None:
        app = _make_app(_quiet_console())
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/result", json={"model": "gpt-4", "result": "done"}
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["ok"] is True

    async def test_rejects_invalid_json(self) -> None:
        app = _make_app(_quiet_console())
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/result",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
            data = await resp.json()
            assert data["ok"] is False

    async def test_renders_model_name(self) -> None:
        out = StringIO()
        console = Console(file=out, highlight=False)
        app = _make_app(console)
        async with TestClient(TestServer(app)) as client:
            await client.post("/result", json={"model": "mymodel", "result": "hi"})
        assert "mymodel" in out.getvalue()

    async def test_missing_fields_use_defaults(self) -> None:
        app = _make_app(_quiet_console())
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/result", json={})
            assert resp.status == 200
