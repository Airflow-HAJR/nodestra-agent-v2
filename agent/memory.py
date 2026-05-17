from supermemory import Supermemory
import re

from agent.config import SUPERMEMORY_API_KEY

_client = Supermemory(api_key=SUPERMEMORY_API_KEY)


def _value(obj, key: str, default=None):
    """Read a key from either a dict-like result or an SDK object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _sanitize_container_tag(raw: str) -> str:
    """Normalize Supermemory container tags to allowed characters."""
    tag = (raw or "").strip()
    # Supermemory expects: alphanumeric, underscore, hyphen, colon
    tag = re.sub(r"[^a-zA-Z0-9_:-]", "", tag)
    return tag


def search_memories(user_id: str, query: str) -> str:
    """Query Supermemory and return only semantically relevant memories."""
    try:
        container_tag = _sanitize_container_tag(user_id)
        if not container_tag:
            return "No relevant user memories found."
        # Semantic search with limit + threshold only (no profile dump)
        search_resp = _client.search.memories(
            q=query,
            container_tag=container_tag,
            search_mode="memories",
            limit=10,
            threshold=0.6,
        )
        results = getattr(search_resp, "results", []) or []
        relevant = []
        for r in results:
            memory = _value(r, "memory", "")
            if memory:
                relevant.append(str(memory))
        if relevant:
            return "Relevant memories:\n" + "\n".join(f"- {m}" for m in relevant)
        return "No relevant user memories found."
    except Exception as e:
        print(f"[MEMORY] search failed: {e}")
        return "No relevant user memories found."


def save_conversation(user_id: str, messages: list) -> None:
    """Save a completed conversation so Supermemory can extract persistent user facts."""
    from langchain_core.messages import AIMessage, HumanMessage

    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"user: {content}")
        elif isinstance(msg, AIMessage) and msg.content:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"assistant: {content}")

    if not lines:
        return

    try:
        container_tag = _sanitize_container_tag(user_id)
        if not container_tag:
            print("[MEMORY] save skipped: invalid empty container tag after sanitization")
            return
        _client.add(content="\n".join(lines), container_tag=container_tag)
        print(f"[MEMORY] saved {len(lines)} turns for user {container_tag}")
    except Exception as e:
        print(f"[MEMORY] save failed: {e}")
