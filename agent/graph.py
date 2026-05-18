import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.llm import build_llm
from agent.memory import get_last_flight, get_last_location, save_conversation, search_memories
from agent.moss_search import search_moss_pois as moss_semantic_search_pois
from agent.prompts import build_system_prompt
from agent.state import State
from agent.timing import add_llm, add_tool
from agent.tools import TOOLS

_TOOL_SPEAK_MESSAGES: dict[str, list[str]] = {
    "get_route": [
        "Calculating the best route for you.",
        "Finding the fastest way to get there.",
        "Mapping your route now.",
    ],
    "find_poi": [
        "Looking that up on the map.",
        "Searching for that location.",
        "Finding that spot for you.",
    ],
    "find_nearest": [
        "Finding the closest one nearby.",
        "Searching what's closest to you.",
        "Looking for the nearest option.",
    ],
    "resolve_poi": [
        "Let me figure out which one you mean.",
        "Checking which floor that's on.",
    ],
    "search_user_memory": [
        "Let me check your preferences.",
        "Looking up what I know about you.",
    ],
    "get_nodes": [
        "Pulling up the airport map.",
        "Loading the map for you.",
    ],
    "set_nav_state": [],  # silent — just a state update
}
_TOOL_SPEAK_PRIORITY = ["get_route", "find_nearest", "find_poi", "resolve_poi", "search_user_memory", "get_nodes"]
_FALLBACK_SPEAK_MESSAGES = [
    "Let me check that for you.",
    "One moment while I look that up.",
]
_EARLY_SPEAK_IDX = 0
_spoken_early_var: ContextVar[bool] = ContextVar("spoken_early", default=False)
_speak_early_callback_var: ContextVar[Callable[[str], None] | None] = ContextVar(
    "speak_early_callback",
    default=None,
)

# Module-level flag: True while tools are executing. External voice interfaces
# (AgentPhone etc.) can poll this to gate STT input.
tools_running: bool = False


def speak_early(text: str) -> None:
    callback = _speak_early_callback_var.get()
    if callback:
        callback(text)
    else:
        print(f"[SPEAK EARLY] {text}")


@contextmanager
def bind_speak_early_callback(callback: Callable[[str], None] | None) -> Iterator[None]:
    token = _speak_early_callback_var.set(callback)
    try:
        yield
    finally:
        _speak_early_callback_var.reset(token)


def reset_turn_state() -> None:
    _spoken_early_var.set(False)


ITERATION_CAP = 30

_llm_with_tools = build_llm().bind_tools(TOOLS)


# ================= phase agent =================

def _make_agent(phase: str):
    def agent(state: State):
        system = build_system_prompt(state, phase=phase)
        print(f"[AGENT:{phase}] thinking...")
        t0 = time.time()

        response = _llm_with_tools.invoke([
            SystemMessage(content=system),
            *state["messages"],
        ])

        dt = time.time() - t0
        add_llm(dt)
        if response.tool_calls:
            names = [tc["name"] if isinstance(tc, dict) else tc.name for tc in response.tool_calls]
            print(f"[AGENT:{phase}] {dt:.2f}s -> calling: {', '.join(names)}")
        else:
            content = response.content
            text = content if isinstance(content, str) else str(content)
            preview = text.replace("\n", " ")[:120]
            print(f"[AGENT:{phase}] {dt:.2f}s -> reply: {preview}")

        return {"messages": [response], "last_error": None}

    return agent


# ================= error handling =================

def _check_tool_errors(state: State) -> Literal["repair", "agent"]:
    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            break
        if getattr(msg, "status", None) == "error":
            return "repair"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if content.strip().lower().startswith("error"):
            return "repair"
    return "agent"


def _repair_node(state: State):
    err = "Unknown tool failure"
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            err = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    print(f"[REPAIR] {err[:200]}")
    return {"last_error": err[:300]}


# ================= commerce prefetch =================

_FOOD_SIGNALS = {
    "halal", "vegetarian", "vegan", "kosher", "gluten-free", "gluten free",
    "chicken", "fish", "seafood", "beef", "italian", "chinese", "japanese",
    "mexican", "indian", "thai", "mediterranean", "sushi", "pizza", "burger",
    "sandwich", "salad", "coffee", "cafe", "bakery",
}
_PAYMENT_SIGNALS = {
    "american express", "amex", "chase sapphire", "chase", "visa", "mastercard",
    "apple pay", "google pay", "priority pass", "dragon pass", "capital one",
}


def _extract_moss_signals(prefs: str) -> str:
    """Pull clean food + payment keywords out of Supermemory bullet text so the
    Moss query is focused signals, not a raw blob of preference prose."""
    lower = prefs.lower()
    hits: list[str] = []
    for kw in _FOOD_SIGNALS:
        if kw in lower:
            hits.append(kw)
    for kw in _PAYMENT_SIGNALS:
        if kw in lower:
            hits.append(kw)
    return " ".join(hits) if hits else "food restaurant"


_COMMERCE_KEYWORDS = {
    "restaurant", "cafe", "coffee", "food", "eat", "hungry", "drink", "bar",
    "lounge", "shop", "store", "duty-free", "dutyfree", "newsstand", "buy",
    "purchase", "pay", "snack", "lunch", "dinner", "breakfast", "meal",
    "sandwich", "pizza", "sushi", "burger", "salad", "bakery", "juice",
    "smoothie", "beer", "wine", "cocktail", "market", "kiosk",
}


def _has_commerce_intent(state: State) -> bool:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            text = (msg.content if isinstance(msg.content, str) else str(msg.content)).lower()
            return any(kw in text for kw in _COMMERCE_KEYWORDS)
    return False


def _commerce_prefetch_node(state: State) -> dict:
    """Enforce the commerce workflow: when the user's message has commerce intent,
    pre-fetch (1) payment preferences from Supermemory, then (2) use those preferences
    to build a richer Moss query so results are already filtered by what the user likes.
    Results land in state.commerce_context so the system prompt injects them directly —
    the LLM never has a chance to skip either call."""
    if not _has_commerce_intent(state):
        return {"commerce_context": None}

    print("[COMMERCE] commerce intent detected — prefetching preferences and POIs")
    t0 = time.time()

    parts: list[str] = []
    prefs: str | None = None

    # Step 1: payment + food preferences from Supermemory
    user_id = state.get("user_id")
    if user_id:
        print(f"[TOOL] search_user_memory query='food preferences dietary restrictions payment cards Apple Pay loyalty programs'")
        t1 = time.time()
        prefs = search_memories(
            user_id,
            "food preferences dietary restrictions payment cards Apple Pay loyalty programs",
        )
        print(f"  -> search_user_memory in {time.time() - t1:.2f}s")
        print(f"[MEMORY] result:\n{prefs}")
        if prefs and "no memories" not in prefs.lower():
            parts.append(f"PAYMENT_PREFERENCES:\n{prefs}")

    # Step 2: extract clean signals from prefs and drive the Moss query with them
    signals = _extract_moss_signals(prefs) if prefs else "food restaurant"
    moss_query = f"restaurant food {signals}"
    print(f"[TOOL] search_moss_pois query={moss_query!r}")
    t2 = time.time()
    pois = moss_semantic_search_pois(query=moss_query, top_k=5)
    print(f"  -> search_moss_pois in {time.time() - t2:.2f}s")
    print(f"[MOSS] pois_result:\n{json.dumps(pois, ensure_ascii=True, indent=2)}")
    if pois:
        parts.append(f"COMMERCE_POIS:\n{json.dumps(pois, ensure_ascii=True)}")

    print(f"[COMMERCE] prefetch done in {time.time() - t0:.2f}s — {len(parts)} section(s)")
    return {"commerce_context": "\n\n".join(parts) if parts else None}


# ================= subgraph builder =================

_tool_node = ToolNode(TOOLS, handle_tool_errors=True)


def _pick_early_phrase(state: State) -> str | None:
    global _EARLY_SPEAK_IDX
    # Find which tools are about to run from the last AI message
    tool_names: list[str] = []
    for msg in reversed(state["messages"]):
        calls = getattr(msg, "tool_calls", None)
        if calls:
            tool_names = [
                (tc["name"] if isinstance(tc, dict) else tc.name) for tc in calls
            ]
            break

    # Pick the highest-priority tool that has phrases
    for tool_name in _TOOL_SPEAK_PRIORITY:
        if tool_name in tool_names:
            phrases = _TOOL_SPEAK_MESSAGES[tool_name]
            if phrases:
                phrase = phrases[_EARLY_SPEAK_IDX % len(phrases)]
                _EARLY_SPEAK_IDX += 1
                return phrase
            return None  # tool explicitly silenced (set_nav_state)

    # Fallback for unknown tools
    phrase = _FALLBACK_SPEAK_MESSAGES[_EARLY_SPEAK_IDX % len(_FALLBACK_SPEAK_MESSAGES)]
    _EARLY_SPEAK_IDX += 1
    return phrase


def _timed_tools(state: State):
    global tools_running

    if not _spoken_early_var.get():
        early_msg_text = _pick_early_phrase(state)
        _spoken_early_var.set(True)
        if early_msg_text:
            speak_early(early_msg_text)

    tools_running = True
    t0 = time.time()
    try:
        result = _tool_node.invoke(state)
    finally:
        tools_running = False
    add_tool(time.time() - t0)
    return result


def _build_subgraph(phase: str):
    builder = StateGraph(State)
    builder.add_node("commerce_prefetch", _commerce_prefetch_node)
    builder.add_node("agent", _make_agent(phase))
    builder.add_node("tools", _timed_tools)
    builder.add_node("repair", _repair_node)
    builder.add_edge(START, "commerce_prefetch")
    builder.add_edge("commerce_prefetch", "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_conditional_edges(
        "tools", _check_tool_errors, {"repair": "repair", "agent": "agent"}
    )
    builder.add_edge("repair", "agent")
    return builder.compile()


# ================= closure (heuristic, no LLM) =================

_DECLINE_PHRASES = ("no thanks", "that's all", "i'm good", "bye", "goodbye", "no i'm set", "nope", "nah", "i'm set", "thank you", "thanks bye")


def _closure_node(state: State) -> dict:
    """Check if the user's latest message is a decline. If so, save conversation and set should_end."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            lower = (msg.content if isinstance(msg.content, str) else str(msg.content)).lower().strip()
            if any(lower == p or lower.startswith(p) for p in _DECLINE_PHRASES):
                user_id = state.get("user_id")
                if user_id:
                    save_conversation(user_id, state["messages"])
                return {"should_end": True}
            break
    return {}


# ================= init (location pre-load) =================

def _init_node(state: State) -> dict:
    """At conversation start, restore the user's last known location and flight from Supermemory."""
    user_id = state.get("user_id")
    if not user_id:
        return {}
    result: dict = {}
    if not state.get("current_location"):
        poi_id = get_last_location(user_id)
        if poi_id:
            print(f"[INIT] restored location from Supermemory: {poi_id}")
            result["current_location"] = poi_id
    if not state.get("flight_number"):
        flight = get_last_flight(user_id)
        if flight:
            print(f"[INIT] restored flight from Supermemory: {flight}")
            result["flight_number"] = flight
    return result


# ================= main graph =================

def _entry_router(state: State) -> Literal["clarify", "navigate"]:
    if not state.get("final_destination"):
        return "clarify"
    return "navigate"


def build_graph():
    clarify = _build_subgraph("clarify")
    navigate = _build_subgraph("navigate")

    builder = StateGraph(State)
    builder.add_node("init", _init_node)
    builder.add_node("clarify", clarify)
    builder.add_node("navigate", navigate)
    builder.add_node("closure", _closure_node)

    builder.add_edge(START, "init")
    builder.add_conditional_edges("init", _entry_router, {"clarify": "clarify", "navigate": "navigate"})

    builder.add_edge("clarify", "closure")
    builder.add_edge("navigate", "closure")
    builder.add_edge("closure", END)

    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()
