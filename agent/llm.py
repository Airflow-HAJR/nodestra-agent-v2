"""Chat model factory.

Two tiers:
  build_llm()            -> capable model for the agent (the part that matters)
  build_llm(fast=True)   -> small/fast model for helper nodes (intent, recall,
                            save, summary) where quality matters less than speed

Plus build_embeddings() for the POI retrieval index (agent/poi_rag.py).

Provider is chosen by LLM_PROVIDER ("openai" default, or "azure").
"""
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from agent.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL_FAST,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MAIN_REASONING_EFFORT,
    OPENAI_MODEL_FAST,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL_MAIN,
)

_embeddings = None


def embedding_model_name() -> str:
    """Which embedding model retrieval will use, or "" if none is configured.

    Returned (rather than inferred inside build_embeddings) because the POI
    index cache is keyed by it: swapping models has to invalidate vectors that
    are no longer comparable."""
    if LLM_PROVIDER == "azure":
        return AZURE_OPENAI_EMBEDDING_DEPLOYMENT if AZURE_OPENAI_API_KEY else ""
    return OPENAI_EMBEDDING_MODEL if OPENAI_API_KEY else ""


def build_embeddings():
    """Embedding model for POI retrieval, or None when none is configured —
    retrieval falls back to keyword matching rather than failing."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    if not embedding_model_name():
        return None
    if LLM_PROVIDER == "azure":
        _embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=SecretStr(AZURE_OPENAI_API_KEY),
            api_version=AZURE_OPENAI_API_VERSION,
        )
    else:
        _embeddings = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=SecretStr(OPENAI_API_KEY),
        )
    return _embeddings


def build_llm(fast: bool = False):
    """Return a chat model. fast=True picks the small/fast model for helper nodes.

    The fast tier prefers Groq (Llama on LPU, ~100-250ms) when GROQ_API_KEY is set,
    since the helper nodes only classify or verbalize — no deep reasoning — and
    Groq's OpenAI-compatible endpoint means it's a drop-in base_url swap. Falls
    back to the OpenAI fast model when no Groq key is configured.
    """
    if LLM_PROVIDER == "azure":
        # Azure exposes one deployment per config; both tiers use it. When that
        # deployment is a reasoning model (gpt-5.6-* like Luna), it rejects
        # function tools on chat completions unless reasoning_effort is set — and
        # the agent binds TOOLS — so pass it through exactly as the OpenAI path
        # does. Left empty for non-reasoning deployments (e.g. gpt-4o), which
        # would 400 on the param.
        azure_kwargs: dict = {}
        if OPENAI_MAIN_REASONING_EFFORT:
            azure_kwargs["reasoning_effort"] = OPENAI_MAIN_REASONING_EFFORT
        return AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=SecretStr(AZURE_OPENAI_API_KEY),
            api_version=AZURE_OPENAI_API_VERSION,
            **azure_kwargs,
        )
    if fast and GROQ_API_KEY:
        return ChatOpenAI(
            model=GROQ_MODEL_FAST,
            api_key=SecretStr(GROQ_API_KEY),
            base_url=GROQ_BASE_URL,
        )
    if fast:
        return ChatOpenAI(
            model=OPENAI_MODEL_FAST,
            api_key=SecretStr(OPENAI_API_KEY),
        )

    # Main tier — the tool-calling agent. Reasoning models (gpt-5.6-*) need
    # reasoning_effort set to bind function tools on chat completions; non-
    # reasoning models (gpt-4o) must NOT receive the param, so it's only passed
    # when explicitly configured.
    main_kwargs: dict = {}
    if OPENAI_MAIN_REASONING_EFFORT:
        main_kwargs["reasoning_effort"] = OPENAI_MAIN_REASONING_EFFORT
    return ChatOpenAI(
        model=OPENAI_MODEL_MAIN,
        api_key=SecretStr(OPENAI_API_KEY),
        **main_kwargs,
    )
