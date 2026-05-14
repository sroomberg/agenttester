"""LLM provider abstractions and implementations."""

from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod


class Provider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def call(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        timeout: int = 120,
    ) -> str: ...


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool definitions to Anthropic format."""
    result = []
    for t in tools:
        fn = t.get("function", {})
        result.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        )
    return result


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-format messages to Anthropic format.

    Groups consecutive ``role: tool`` messages into a single user message
    with ``tool_result`` content blocks, and converts assistant ``tool_calls``
    to ``tool_use`` content blocks.
    """
    result: list[dict] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "tool":
            tool_results: list[dict] = []
            while i < len(messages) and messages[i]["role"] == "tool":
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": messages[i]["tool_call_id"],
                        "content": messages[i]["content"],
                    }
                )
                i += 1
            result.append({"role": "user", "content": tool_results})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            content: list[dict] = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": fn["name"],
                        "input": args,
                    }
                )
            result.append({"role": "assistant", "content": content})
            i += 1
        else:
            result.append({"role": m["role"], "content": m["content"]})
            i += 1
    return result


def _from_anthropic_response(data: dict) -> dict:
    """Convert an Anthropic Messages API response to a normalized dict.

    Returns ``{"content": str | None, "tool_calls": list | None}`` matching
    the shape produced by ``OpenAICompatProvider.call_raw``.
    """
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block["id"],
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )
    return {
        "content": "\n".join(text_parts) if text_parts else None,
        "tool_calls": tool_calls if tool_calls else None,
    }


class AnthropicProvider(Provider):
    """Calls the Anthropic Messages API directly."""

    def __init__(self, api_key_env: str = "ANTHROPIC_API_KEY") -> None:
        self.api_key_env = api_key_env

    def call_raw(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        timeout: int = 120,
        tools: list[dict] | None = None,
    ) -> dict:
        """Single API call with optional tool use.

        Accepts *messages* and *tools* in OpenAI format; converts to Anthropic
        format before sending.  Returns a normalized dict matching the shape of
        ``OpenAICompatProvider.call_raw``: ``{"content": …, "tool_calls": …}``.
        """
        api_key = os.environ.get(self.api_key_env, "")
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _to_anthropic_messages(
                [m for m in messages if m["role"] != "system"]
            ),
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if tools:
            body["tools"] = _to_anthropic_tools(tools)
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return _from_anthropic_response(data)

    def call(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        timeout: int = 120,
    ) -> str:
        return self.call_raw(model, messages, max_tokens, timeout).get("content") or ""


class OpenAICompatProvider(Provider):
    """Calls any OpenAI-compatible chat completions endpoint.

    Covers vLLM, Azure AI Foundry, GCP Vertex, and similar services.
    Sends an ``Authorization: Bearer`` header when *api_key_env* is set.
    """

    def __init__(self, endpoint: str, api_key_env: str | None = None) -> None:
        self.endpoint = endpoint
        self.api_key_env = api_key_env

    def call_raw(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        timeout: int = 120,
        tools: list[dict] | None = None,
    ) -> dict:
        """Single API call. Returns the full response message dict.

        May contain ``tool_calls`` when *tools* are provided and the model
        chooses to invoke one.
        """
        api_key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body: dict = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            f"{self.endpoint.rstrip('/')}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]

    def call(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        timeout: int = 120,
    ) -> str:
        return self.call_raw(model, messages, max_tokens, timeout).get("content") or ""


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

    def _make_client(self, timeout: int):
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise ImportError(
                "boto3 is required for AWS Bedrock: pip install agenttester[aws]"
            ) from None

        config = Config(connect_timeout=timeout, read_timeout=timeout)

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
