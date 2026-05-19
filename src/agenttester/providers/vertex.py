"""GCP Vertex AI provider."""

from __future__ import annotations

from .base import _fetch_cli_token
from .openai_compat import OpenAICompatProvider

_VERTEX_CLI_CMD = "gcloud auth print-access-token"


class VertexProvider(OpenAICompatProvider):
    """GCP Vertex AI (OpenAI-compatible endpoint).

    Two auth modes:

    ``auth_method='api_key'`` (default): reads ``api_key_env`` from the
    environment and sends it as an ``Authorization: Bearer`` header (identical
    to the generic ``openai`` provider type).

    ``auth_method='cli'``: runs ``gcloud auth print-access-token`` to obtain a
    short-lived access token. The token is cached for 55 minutes. Requires the
    Google Cloud SDK to be installed and ``gcloud auth login`` to have been run.

    Config example::

        providers:
          my-vertex:
            type: vertex
            endpoint: https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-project/locations/us-central1/endpoints/openapi
            auth_method: cli            # or "api_key"
            api_key_env: VERTEX_TOKEN   # only needed for auth_method: api_key
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
        if self._auth_method == "cli":
            token = _fetch_cli_token(_VERTEX_CLI_CMD)
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
        return super()._headers()
