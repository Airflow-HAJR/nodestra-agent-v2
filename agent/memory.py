from supermemory import Supermemory

from agent.config import SUPERMEMORY_API_KEY

_client = Supermemory(api_key=SUPERMEMORY_API_KEY)


def search_memories(user_id: str, query: str) -> str:
    """Query Supermemory for user context relevant to the given query."""
    try:
        profile = _client.profile(container_tag=user_id, q=query)

        parts = []

        static = getattr(getattr(profile, "profile", None), "static", []) or []
        dynamic = getattr(getattr(profile, "profile", None), "dynamic", []) or []
        results = getattr(getattr(profile, "search_results", None), "results", []) or []

        if static:
            parts.append("Known about this user:\n" + "\n".join(f"- {s}" for s in static))
        if dynamic:
            parts.append("Recent context:\n" + "\n".join(f"- {d}" for d in dynamic))
        relevant = [r.get("memory", "") for r in results if r.get("memory")]
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
