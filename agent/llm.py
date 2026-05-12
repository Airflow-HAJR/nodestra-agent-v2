from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agent.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
)


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=AZURE_OPENAI_DEPLOYMENT,
        base_url=AZURE_OPENAI_ENDPOINT.rstrip("/") + "/openai/v1/",
        api_key=SecretStr(AZURE_OPENAI_API_KEY),
        default_query={"api-version": "preview"},
    )
