"""Tests for Cursor CLI usage parsing (run + REPL shared helpers)."""

from __future__ import annotations

import json

from agenttester.cursor_usage import (
    CursorStreamParser,
    detect_cursor_output_format,
    parse_cursor_json_stdout,
    usage_from_payload,
)


class TestDetectCursorOutputFormat:
    def test_stream_json_flag(self) -> None:
        cmd = "agent -p --output-format stream-json --stream-partial-output hi"
        assert detect_cursor_output_format(cmd) == "stream-json"

    def test_json_flag(self) -> None:
        cmd = "agent -p --output-format json {prompt}"
        assert detect_cursor_output_format(cmd) == "json"

    def test_json_equals_form(self) -> None:
        assert detect_cursor_output_format("agent --output-format=json hi") == "json"

    def test_none_when_plain(self) -> None:
        assert detect_cursor_output_format("agent -p --force {prompt}") is None


class TestUsageFromPayload:
    def test_sums_cache_fields(self) -> None:
        usage = usage_from_payload(
            {
                "usage": {
                    "inputTokens": 7,
                    "cacheReadTokens": 100,
                    "cacheWriteTokens": 20,
                    "outputTokens": 50,
                }
            }
        )
        assert usage.total_input == 127
        assert usage.output == 50
        assert usage.input == 7
        assert usage.cache_read == 100
        assert usage.cache_write == 20

    def test_snake_case_fields(self) -> None:
        usage = usage_from_payload(
            {
                "usage": {
                    "input_tokens": 10,
                    "cache_read_tokens": 5,
                    "output_tokens": 3,
                }
            }
        )
        assert usage.total_input == 15
        assert usage.output == 3

    def test_cost_usd(self) -> None:
        usage = usage_from_payload({"usage": {"inputTokens": 1, "costUsd": 0.0123}})
        assert usage.cost_usd == 0.0123

    def test_empty_usage(self) -> None:
        usage = usage_from_payload({})
        assert not usage.has_usage


class TestCursorStreamParser:
    def _events(self, *events: dict) -> CursorStreamParser:
        parser = CursorStreamParser()
        for event in events:
            parser.process_line(json.dumps(event))
        return parser

    def test_partial_assistant_stream(self) -> None:
        parser = CursorStreamParser()
        chunks: list[str | None] = []
        for text in ("Hel", "lo"):
            chunks.append(
                parser.process_line(
                    json.dumps(
                        {
                            "type": "assistant",
                            "timestamp_ms": 1,
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": text}],
                            },
                        }
                    )
                )
            )
        assert chunks == ["Hel", "lo"]
        assert not parser.usage.has_usage

    def test_result_accumulates_usage(self) -> None:
        parser = self._events(
            {
                "type": "result",
                "result": "done",
                "usage": {
                    "inputTokens": 10,
                    "cacheReadTokens": 5,
                    "outputTokens": 20,
                },
            }
        )
        assert parser.usage.total_input == 15
        assert parser.usage.output == 20

    def test_non_json_line_passthrough(self) -> None:
        parser = CursorStreamParser()
        assert parser.process_line("plain log line") == "plain log line"


class TestParseCursorJsonStdout:
    def test_parses_result_and_usage(self) -> None:
        payload = json.dumps(
            {
                "result": "Hello",
                "usage": {"inputTokens": 3, "outputTokens": 7},
            }
        )
        text, usage = parse_cursor_json_stdout(payload)
        assert text == "Hello"
        assert usage.total_input == 3
        assert usage.output == 7

    def test_invalid_json_returns_raw(self) -> None:
        text, usage = parse_cursor_json_stdout("not json")
        assert text == "not json"
        assert not usage.has_usage
