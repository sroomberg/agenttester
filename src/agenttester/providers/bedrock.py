"""AWS Bedrock Converse API provider."""

from __future__ import annotations

import os

from .base import Provider


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

    async def async_call(
        self, model: str, messages: list[dict], max_tokens: int
    ) -> str:
        import asyncio

        return await asyncio.to_thread(self.call, model, messages, max_tokens)
