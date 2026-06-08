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
    tag = (raw or "").strip().lstrip("+")
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
    except Exception:
        return "No relevant user memories found."


def _location_custom_id(container_tag: str) -> str:
    return f"location:{container_tag}"


def _flight_custom_id(container_tag: str) -> str:
    return f"flight:{container_tag}"


def get_last_location(user_id: str) -> str | None:
    """Return the last known POI id for this user, or None if not stored."""
    try:
        container_tag = _sanitize_container_tag(user_id)
        if not container_tag:
            return None
        custom_id = _location_custom_id(container_tag)
        resp = _client.documents.list(
            container_tags=[container_tag],
            include_content=True,
        )
        docs = getattr(resp, "memories", []) or []
        for doc in docs:
            if _value(doc, "custom_id", None) == custom_id:
                content = _value(doc, "content", "") or ""
                for line in content.splitlines():
                    if line.startswith("poi_id:"):
                        return line[len("poi_id:"):].strip()
        return None
    except Exception:
        return None


def update_location(user_id: str, poi_id: str, poi_name: str) -> None:
    """Overwrite the user's current location — delete all prior location docs first so
    only one location state ever exists in Supermemory."""
    try:
        container_tag = _sanitize_container_tag(user_id)
        if not container_tag:
            return
        custom_id = _location_custom_id(container_tag)
        content = f"poi_id:{poi_id}\npoi_name:{poi_name}"

        # Delete every existing location doc (custom_id match OR legacy poi_id: content)
        # before writing the new one so stale states can't accumulate.
        resp = _client.documents.list(container_tags=[container_tag], include_content=True)
        docs = getattr(resp, "memories", []) or []
        for doc in docs:
            is_location_doc = (
                _value(doc, "custom_id", None) == custom_id
                or "poi_id:" in (_value(doc, "content", "") or "")
            )
            if is_location_doc:
                doc_id = _value(doc, "id", None)
                if doc_id:
                    try:
                        _client.documents.delete(id=doc_id)
                    except Exception:
                        pass

        _client.documents.add(
            content=content,
            container_tag=container_tag,
            custom_id=custom_id,
        )
    except Exception:
        pass


def get_last_flight(user_id: str) -> str | None:
    """Return the last known flight number for this user, or None if not stored."""
    try:
        container_tag = _sanitize_container_tag(user_id)
        if not container_tag:
            return None
        custom_id = _flight_custom_id(container_tag)
        resp = _client.documents.list(container_tags=[container_tag], include_content=True)
        docs = getattr(resp, "memories", []) or []
        for doc in docs:
            if _value(doc, "custom_id", None) == custom_id:
                content = _value(doc, "content", "") or ""
                for line in content.splitlines():
                    if line.startswith("flight_number:"):
                        return line[len("flight_number:"):].strip()
        return None
    except Exception:
        return None


def update_flight(user_id: str, flight_number: str) -> None:
    """Upsert the user's current flight number as a Supermemory dynamic document."""
    try:
        container_tag = _sanitize_container_tag(user_id)
        if not container_tag:
            return
        custom_id = _flight_custom_id(container_tag)
        content = f"flight_number:{flight_number}"
        resp = _client.documents.list(container_tags=[container_tag], include_content=True)
        docs = getattr(resp, "memories", []) or []
        for doc in docs:
            if _value(doc, "custom_id", None) == custom_id:
                doc_id = _value(doc, "id", None)
                if doc_id:
                    _client.documents.update(id=doc_id, content=content)
                    return
        _client.documents.add(content=content, container_tag=container_tag, custom_id=custom_id)
    except Exception:
        pass


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
            return
        _client.add(content="\n".join(lines), container_tag=container_tag)
    except Exception:
        pass
