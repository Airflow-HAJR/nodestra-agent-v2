"""
Shared Azure OpenAI embedding client.

vector_memory (user memories) embeds text through here. Single client, single
"deployment missing" latch — if the embedding deployment is unavailable we stop
retrying and callers fall back to non-vector behavior.
"""
from __future__ import annotations

import logging

from agent.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
)

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1536

_client = None
_broken = False  # latched True on DeploymentNotFound so we stop retrying


def embeddings_available() -> bool:
    return not _broken


def embed(text: str) -> list[float]:
    """Return the embedding vector for *text*. Raises if the deployment is unavailable."""
    global _client, _broken
    if _broken:
        raise RuntimeError("embedding deployment unavailable")
    if _client is None:
        from openai import AzureOpenAI
        _client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    try:
        resp = _client.embeddings.create(
            input=text,
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        )
        return resp.data[0].embedding
    except Exception as e:
        if "DeploymentNotFound" in str(e) or "404" in str(e):
            _broken = True
            logger.warning(
                "Embedding deployment '%s' not found — vector features disabled. "
                "Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT to a deployed model name.",
                AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            )
        raise
