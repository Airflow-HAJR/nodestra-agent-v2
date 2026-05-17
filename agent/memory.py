from supermemory import Supermemory

from agent.config import SUPERMEMORY_API_KEY

_client = Supermemory(api_key=SUPERMEMORY_API_KEY)


def _value(obj, key: str, default=None):
    """Read a key from either a dict-like result or an SDK object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def search_memories(user_id: str, query: str) -> str:
    """Query Supermemory for user context relevant to the given query."""
    try:
        parts = []

        # Static/dynamic user profile (no query needed)
        profile = _client.profile(container_tag=user_id)
        static = getattr(getattr(profile, "profile", None), "static", []) or []
        dynamic = getattr(getattr(profile, "profile", None), "dynamic", []) or []
        if static:
            parts.append("Known about this user:\n" + "\n".join(f"- {s}" for s in static))
        if dynamic:
            parts.append("Recent context:\n" + "\n".join(f"- {d}" for d in dynamic))

        # Semantic search with limit + threshold
        search_resp = _client.search.memories(
            q=query,
            container_tag=user_id,
            search_mode="memories",
            limit=5,
            threshold=0.6,
        )
        results = getattr(search_resp, "results", []) or []
        relevant = []
        for r in results:
            memory = _value(r, "memory", "")
            if memory:
                relevant.append(str(memory))
        if relevant:
            parts.append("Relevant memories:\n" + "\n".join(f"- {m}" for m in relevant))

        return "\n\n".join(parts) if parts else "No relevant user memories found."
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
        _client.add(content="\n".join(lines), container_tag=user_id)
        print(f"[MEMORY] saved {len(lines)} turns for user {user_id}")
    except Exception as e:
        print(f"[MEMORY] save failed: {e}")
