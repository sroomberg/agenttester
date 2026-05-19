"""Azure AI Foundry / Azure OpenAI Service provider."""

from __future__ import annotations

import os

from .base import _fetch_cli_token
from .openai_compat import OpenAICompatProvider

_AZURE_CLI_CMD = (
    "az account get-access-token "
    "--resource https://cognitiveservices.azure.com "
    "--query accessToken -o tsv"
)


class AzureProvider(OpenAICompatProvider):
    """Azure AI Foundry / Azure OpenAI Service.

    Two auth modes:

    ``auth_method='api_key'`` (default): reads ``api_key_env`` from the
    environment and sends it as an ``api-key`` header (the Azure OpenAI
    key-based auth scheme).

    ``auth_method='cli'``: runs ``az account get-access-token`` to obtain an
    Entra ID Bearer token. The token is cached for 55 minutes. Requires the
    Azure CLI to be installed and ``az login`` to have been run.

    Config example::

        providers:
          my-azure:
            type: azure
            endpoint: https://my-resource.openai.azure.com
            auth_method: api_key        # or "cli"
            api_key_env: AZURE_OPENAI_KEY
    """

    def __init__(
        self,
        endpoint: str,
        api_key_env: str | None = None,
        auth_method: str = "api_key",
    ) -> None:
        super().__init__(endpoint, api_key_env)
        self._auth_method = auth_method

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._auth_method == "cli":
            token = _fetch_cli_token(_AZURE_CLI_CMD)
            headers["Authorization"] = f"Bearer {token}"
        else:
            api_key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
            if api_key:
                headers["api-key"] = api_key
        return headers
