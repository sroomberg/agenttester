"""AWS Bedrock Converse API provider."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable

from .base import Provider


def _to_bedrock_tools(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool definitions to Bedrock Converse format."""
    result = []
    for t in tools:
        fn = t.get("function", {})
        result.append(
            {
                "toolSpec": {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "inputSchema": {
                        "json": fn.get(
                            "parameters", {"type": "object", "properties": {}}
                        )
                    },
                }
            }
        )
    return result


def _to_bedrock_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Convert OpenAI-format messages to Bedrock Converse format.

    Returns (system_prompts, converse_messages).
    """
    system: list[dict] = []
    converse: list[dict] = []

    i = 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "system":
            system.append({"text": m["content"]})
            i += 1
        elif m["role"] == "tool":
            tool_results: list[dict] = []
            while i < len(messages) and messages[i]["role"] == "tool":
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": messages[i]["tool_call_id"],
                            "content": [{"text": messages[i]["content"]}],
                        }
                    }
                )
                i += 1
            converse.append({"role": "user", "content": tool_results})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            content: list[dict] = []
            if m.get("content"):
                content.append({"text": m["content"]})
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                content.append(
                    {
                        "toolUse": {
                            "toolUseId": tc["id"],
                            "name": fn["name"],
                            "input": args,
                        }
                    }
                )
            converse.append({"role": "assistant", "content": content})
            i += 1
        else:
            converse.append({"role": m["role"], "content": [{"text": m["content"]}]})
            i += 1

    return system, converse


class BedrockProvider(Provider):
    """Calls AWS Bedrock via the Converse API using boto3.

    Requires ``pip install agenttester[aws]``.

    Authentication priority (first configured wins):
    1. *aws_profile* — uses a named ``~/.aws/config`` profile (SSO, assumed
       roles, etc.)
    2. *aws_access_key_id_env* / *aws_secret_access_key_env* — reads explicit
       credentials from environment variables.
    3. Default boto3 credential chain (env vars, ``~/.aws/credentials``, IAM
       instance role, etc.).
    """

    def __init__(
        self,
        region: str = "us-east-1",
        aws_profile: str | None = None,
        aws_access_key_id_env: str | None = None,
        aws_secret_access_key_env: str | None = None,
        aws_session_token_env: str | None = None,
    ) -> None:
        self.region = region
        self.aws_profile = aws_profile
        self.aws_access_key_id_env = aws_access_key_id_env
        self.aws_secret_access_key_env = aws_secret_access_key_env
        self.aws_session_token_env = aws_session_token_env

    def _make_client(self, timeout: int = 300):
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise ImportError(
                "boto3 is required for AWS Bedrock: pip install agenttester[aws]"
            ) from None

        config = Config(
            connect_timeout=30,
            read_timeout=timeout,
            retries={"max_attempts": 2},
        )

        if self.aws_profile:
            session = boto3.Session(profile_name=self.aws_profile)
        elif self.aws_access_key_id_env:
            session = boto3.Session(
                aws_access_key_id=os.environ.get(self.aws_access_key_id_env),
                aws_secret_access_key=os.environ.get(
                    self.aws_secret_access_key_env or "", ""
                ),
                aws_session_token=os.environ.get(self.aws_session_token_env)
                if self.aws_session_token_env
                else None,
            )
        else:
            session = boto3.Session()

        return session.client("bedrock-runtime", region_name=self.region, config=config)

    def call(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        timeout: int = 120,
    ) -> str:
        client = self._make_client(timeout)

        system = [{"text": m["content"]} for m in messages if m["role"] == "system"]
        converse_messages = [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in messages
            if m["role"] != "system"
        ]
        kwargs: dict = {
            "modelId": model,
            "messages": converse_messages,
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system:
            kwargs["system"] = system

        response = client.converse(**kwargs)
        return response["output"]["message"]["content"][0]["text"]

    async def async_call(
        self, model: str, messages: list[dict], max_tokens: int
    ) -> str:
        return await asyncio.to_thread(self.call, model, messages, max_tokens)

    def _stream_raw_sync(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        tools: list[dict] | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict:
        """Synchronous streaming call via Bedrock ConverseStream."""
        client = self._make_client()
        system, converse_messages = _to_bedrock_messages(messages)

        kwargs: dict = {
            "modelId": model,
            "messages": converse_messages,
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["toolConfig"] = {"tools": _to_bedrock_tools(tools)}

        response = client.converse_stream(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        current_tool: dict | None = None

        for event in response["stream"]:
            if "contentBlockStart" in event:
                start = event["contentBlockStart"].get("start", {})
                if "toolUse" in start:
                    current_tool = {
                        "id": start["toolUse"]["toolUseId"],
                        "function": {
                            "name": start["toolUse"]["name"],
                            "arguments": "",
                        },
                    }

            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    chunk = delta["text"]
                    text_parts.append(chunk)
                    if on_chunk and chunk:
                        on_chunk(chunk)
                elif "toolUse" in delta and current_tool is not None:
                    current_tool["function"]["arguments"] += delta["toolUse"].get(
                        "input", ""
                    )

            elif "contentBlockStop" in event:
                if current_tool is not None:
                    tool_calls.append(current_tool)
                    current_tool = None

        text = "".join(text_parts)
        return {
            "content": text if text else None,
            "tool_calls": tool_calls if tool_calls else None,
        }

    async def async_stream_raw(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        tools: list[dict] | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict:
        """Async streaming via Bedrock ConverseStream (runs sync in thread)."""
        return await asyncio.to_thread(
            self._stream_raw_sync, model, messages, max_tokens, tools, on_chunk
        )
