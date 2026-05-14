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


class AnthropicProvider(Provider):
    """Calls the Anthropic Messages API directly."""

    def __init__(self, api_key_env: str = "ANTHROPIC_API_KEY") -> None:
        self.api_key_env = api_key_env

    def call(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        timeout: int = 120,
    ) -> str:
        api_key = os.environ.get(self.api_key_env, "")
        payload = json.dumps(
            {"model": model, "max_tokens": max_tokens, "messages": messages}
        ).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return data["content"][0]["text"]


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
