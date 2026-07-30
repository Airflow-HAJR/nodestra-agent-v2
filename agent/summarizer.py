import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a call summarizer. Given a conversation between a user and an airport "
    "navigation assistant, return JSON with: "
    '"summary" (1-2 sentences what happened), '
    '"topics" (list from: gate_query, directions, restroom, flight_status, general_info, unresolved), '
    '"resolved" (true if the user got what they needed). '
    "Output valid JSON only, nothing else."
)

_DEFAULT = {"summary": "", "topics": [], "resolved": False}


def summarize(messages: list, state: dict[str, Any]) -> dict:
    """Run a single cheap LLM call and return {summary, topics, resolved}."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from agent.llm import build_llm

        llm = build_llm(fast=True)

        # Trim to last 20 messages to keep prompt small
        trimmed = messages[-20:] if len(messages) > 20 else messages

        # Serialize messages to plain text
        lines: list[str] = []
        for m in trimmed:
            role = type(m).__name__.replace("Message", "")
            content = m.content if isinstance(m.content, str) else str(m.content)
            if content.strip():
                lines.append(f"{role}: {content.strip()}")

        conversation = "\n".join(lines) or "(empty conversation)"

        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=conversation),
        ])

        raw = response.content if isinstance(response.content, str) else str(response.content)
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        return {
            "summary": str(parsed.get("summary", "")),
            "topics": [str(t) for t in parsed.get("topics", [])],
            "resolved": bool(parsed.get("resolved", False)),
        }
    except Exception:
        logger.exception("summarize failed — using empty default")
        return dict(_DEFAULT)
