"""Chat model factory.

Two tiers:
  build_llm()            -> capable model for the agent (the part that matters)
  build_llm(fast=True)   -> small/fast model for helper nodes (intent, recall,
                            save, summary) where quality matters less than speed

Provider is chosen by LLM_PROVIDER ("openai" default, or "azure").
"""
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import SecretStr

from agent.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL_FAST,
    OPENAI_MODEL_MAIN,
)


def build_llm(fast: bool = False):
    """Return a chat model. fast=True picks the small/fast model for helper nodes."""
    if LLM_PROVIDER == "azure":
        # Azure exposes one deployment per config; both tiers use it.
        return AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=SecretStr(AZURE_OPENAI_API_KEY),
            api_version=AZURE_OPENAI_API_VERSION,
        )
    return ChatOpenAI(
        model=OPENAI_MODEL_FAST if fast else OPENAI_MODEL_MAIN,
        api_key=SecretStr(OPENAI_API_KEY),
    )
