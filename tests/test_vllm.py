"""Tests for agenttester.vllm."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from agenttester.vllm import check_connection


class TestCheckConnection:
    async def test_returns_true_when_reachable(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        with patch("aiohttp.ClientSession", MagicMock(return_value=mock_session)):
            result = await check_connection("http://host:8001")
        assert result is True

    async def test_returns_false_on_exception(self) -> None:
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.side_effect = Exception("refused")
        with patch("aiohttp.ClientSession", MagicMock(return_value=mock_session)):
            result = await check_connection("http://host:8001")
        assert result is False
