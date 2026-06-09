import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import MEMORY_ENABLED
from agent.llm import build_llm
from agent.memory import get_last_flight, get_last_location
from agent.prompts import build_system_prompt
from agent.state import State
from agent.timing import add_llm, add_supermemory, add_tool
from agent.tools import TOOLS
from agent.logger import log_event

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
    "recall_user_memories": [
        "Let me check your preferences.",
        "Looking up what I know about you.",
    ],
    "get_nodes": [
        "Pulling up the airport map.",
        "Loading the map for you.",
    ],
    "set_nav_state": [],   # silent — just a state update
    "end_call": [],        # silent — just a state update
}
_TOOL_SPEAK_PRIORITY = ["get_route", "find_nearest", "find_poi", "resolve_poi", "recall_user_memories", "get_nodes"]
_FALLBACK_SPEAK_MESSAGES = [
    "Let me check that for you.",
    "One moment while I look that up.",
]

# Per-context state — safe for concurrent calls
_spoken_early_var: ContextVar[bool] = ContextVar("spoken_early", default=False)
_speak_early_callback_var: ContextVar[Callable[[str], None] | None] = ContextVar(
    "speak_early_callback",
    default=None,
)
_sentence_callback_var: ContextVar[Callable[[str | None], None] | None] = ContextVar(
    "sentence_callback",
    default=None,
)
_tools_running_var: ContextVar[bool] = ContextVar("tools_running", default=False)
_early_speak_idx_var: ContextVar[int] = ContextVar("early_speak_idx", default=0)

# Module-level list for analytics tracking (see get_turn_tools_used)
_turn_tools_used: list[str] = []


def get_turn_tools_used() -> list[str]:
    """Return tool names used in this turn (for analytics)."""
    return list(_turn_tools_used)


def get_tools_running() -> bool:
    """Per-context getter for STT gating. Each concurrent call has its own value."""
    return _tools_running_var.get()


def speak_early(text: str) -> None:
    callback = _speak_early_callback_var.get()
    if callback:
        callback(text)
    else:
        print("[thinking]")


@contextmanager
def bind_speak_early_callback(callback: Callable[[str], None] | None) -> Iterator[None]:
    token = _speak_early_callback_var.set(callback)
    try:
        yield
    finally:
        _speak_early_callback_var.reset(token)


@contextmanager
def bind_sentence_callback(callback: Callable[[str | None], None] | None) -> Iterator[None]:
    token = _sentence_callback_var.set(callback)
    try:
        yield
    finally:
        _sentence_callback_var.reset(token)


def reset_turn_state() -> None:
    global _turn_tools_used
    _spoken_early_var.set(False)
    _early_speak_idx_var.set(0)
    _turn_tools_used = []


ITERATION_CAP = 30

_llm_with_tools = build_llm().bind_tools(TOOLS)


# ================= agent node =================

def _agent_node(state: State):
    system = build_system_prompt(state)  # phase derived dynamically from state
    print("[thinking]")
    t0 = time.time()

    sentence_cb = _sentence_callback_var.get()

    if sentence_cb is not None:
        # Stream tokens; fire sentence TTS callbacks for final (non-tool-call) responses.
        from functools import reduce
        chunks = []
        buffer = ""
        has_tool_calls = False
        for chunk in _llm_with_tools.stream([SystemMessage(content=system), *state["messages"]]):
            chunks.append(chunk)
            if chunk.tool_calls:
                has_tool_calls = True
            if not has_tool_calls and isinstance(chunk.content, str) and chunk.content:
                buffer += chunk.content
                # Flush complete sentences as they arrive.
                while True:
                    found = False
                    for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
                        idx = buffer.find(sep)
                        if idx >= 0:
                            sentence = buffer[:idx + len(sep)].strip()
                            buffer = buffer[idx + len(sep):]
                            if sentence:
                                sentence_cb(sentence)
                            found = True
                            break
                    if not found:
                        break
        # Flush any remaining text.
        if buffer.strip() and not has_tool_calls:
            sentence_cb(buffer.strip())
        # Signal end of streaming (only for text responses — tool-call turns don't stream TTS).
        if not has_tool_calls:
            sentence_cb(None)
        response = reduce(lambda a, b: a + b, chunks) if chunks else _llm_with_tools.invoke(
            [SystemMessage(content=system), *state["messages"]]
        )
    else:
        # Reset active_intents each turn so stale intents from prior turns don't pollute routing.
        # Commerce prefetch and set_nav_state will repopulate them for this turn.
        response = _llm_with_tools.invoke([
            SystemMessage(content=system),
            *state["messages"],
        ])

    dt = time.time() - t0
    add_llm(dt)

    return {"messages": [response], "last_error": None, "active_intents": []}


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
    print("[thinking]")
    return {"last_error": err[:300]}


# ================= intent detection & prefetch =================

# Per-intent keyword sets for intents that benefit from a Supermemory personalization prefetch.
_INTENT_KEYWORDS: dict[str, set[str]] = {
    "navigate": {
        "gate", "terminal", "get to", "how do i get", "directions", "navigate",
        "take me", "route to", "way to", "walk to", "find my way", "where is",
        "how far", "which way", "i need to go", "need to get to",
    },
    "food_drinks": {
        "restaurant", "cafe", "coffee", "food", "eat", "hungry", "drink", "bar",
        "snack", "lunch", "dinner", "breakfast", "meal", "sandwich", "pizza",
        "sushi", "burger", "salad", "bakery", "juice", "smoothie", "beer",
        "wine", "cocktail", "market", "kiosk", "boba", "tea", "ramen", "tacos",
    },
    "shopping": {
        "shop", "store", "duty-free", "dutyfree", "newsstand", "buy", "purchase",
        "souvenir", "gift", "retail", "boutique", "pharmacy", "drugstore",
        "bookstore", "electronics", "sunglasses", "clothes", "fashion", "perfume",
    },
    "lounge_access": {
        "lounge", "priority pass", "dragon pass", "club", "amex lounge",
        "chase lounge", "centurion", "admirals club", "united club", "sky club",
        "alaska lounge", "first class lounge", "business lounge",
    },
    "payment": {
        "amex", "american express", "visa", "mastercard", "chase sapphire",
        "apple pay", "google pay", "capital one", "card", "cashback", "rewards",
        "miles", "points", "accept", "payment", "tap to pay",
    },
}

# Intents detected purely from message keywords (no prefetch node needed for these).
_KEYWORD_ONLY_INTENTS: dict[str, set[str]] = {
    "flight_info": {
        "flight", "gate", "boarding", "delay", "depart", "arrival", "connection",
        "layover", "terminal", "on time", "cancelled", "status",
    },
    "airport_infrastructure": {
        "tsa", "security", "customs", "immigration", "precheck", "clear",
        "global entry", "liquids", "checkpoint", "screening", "id", "passport",
        "carry-on", "prohibited",
    },
    "ground_transport": {
        "bart", "taxi", "uber", "lyft", "rideshare", "rental car", "shuttle",
        "parking", "bus", "train", "transit", "pickup", "dropoff", "hotel",
    },
    "baggage": {
        "baggage", "luggage", "suitcase", "bag claim", "carousel", "oversized",
        "lost bag", "storage", "locker", "checked bag",
    },
    "accessibility": {
        "wheelchair", "elevator", "accessible", "disability", "mobility",
        "hearing", "visual", "blind", "deaf", "assistance", "ramp",
    },
    "wellness": {
        "spa", "massage", "meditation", "quiet", "chapel", "prayer",
        "nursing", "lactation", "mother", "pet relief", "yoga", "relax",
    },
    "charging_connectivity": {
        "charge", "charging", "outlet", "wifi", "wi-fi", "internet",
        "power", "usb", "plug", "business center", "laptop",
    },
    "family_services": {
        "family", "stroller", "kid", "child", "children", "baby", "toddler",
        "play area", "family restroom",
    },
    "tourism": {
        "art", "exhibit", "museum", "mural", "installation", "architecture",
        "history", "interesting", "explore", "sightseeing", "display",
    },
}


def _detect_intents(text: str) -> list[str]:
    """Return all intents whose keywords appear in text, prefetch intents first."""
    lower = text.lower()
    found: list[str] = []
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            found.append(intent)
    for intent, keywords in _KEYWORD_ONLY_INTENTS.items():
        if any(kw in lower for kw in keywords):
            found.append(intent)
    return found


def _prefetch_router(state: State) -> Literal["prefetch", "agent"]:
    """Route to prefetch only when the message contains a prefetch-eligible intent."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            text = (msg.content if isinstance(msg.content, str) else str(msg.content)).lower()
            if any(any(kw in text for kw in kws) for kws in _INTENT_KEYWORDS.values()):
                return "prefetch"
            return "agent"
    return "agent"


def _prefetch_node(state: State) -> dict:
    """Pre-fetch personalization data for intents where it meaningfully affects the response.

    Fetches Supermemory user preferences for navigate, food_drinks, shopping, lounge_access,
    and payment intents. Sets specific detected intent names in active_intents so PHASE_FOCUS
    renders the right instructions.
    """
    from agent.memory import search_memories

    # Detect which specific intents are active from the latest message
    detected: list[str] = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            detected = _detect_intents(text)
            break

    # Merge with any intents already in state (e.g. navigate set by set_nav_state)
    existing = list(state.get("active_intents") or [])
    merged_intents = list({*existing, *detected})

    if not MEMORY_ENABLED:
        return {"commerce_context": None, "active_intents": merged_intents}

    t0 = time.time()
    parts: list[str] = []
    prefs: str | None = None
    user_id = state.get("user_id")

    active_prefetch = [i for i in detected if i in _INTENT_KEYWORDS]
    needs_navigate = "navigate" in active_prefetch
    needs_food = "food_drinks" in active_prefetch
    needs_lounge = "lounge_access" in active_prefetch
    needs_payment = "payment" in active_prefetch or "shopping" in active_prefetch

    # Supermemory: fetch all relevant preference types in a single query
    if user_id and active_prefetch:
        query_parts = []
        if needs_navigate:
            query_parts.append("accessibility needs wheelchair elevator route preferences mobility")
        if needs_food:
            query_parts.append("food preferences dietary restrictions cuisine")
        if needs_lounge:
            query_parts.append("lounge access priority pass credit card lounge membership")
        if needs_payment:
            query_parts.append("payment cards Apple Pay loyalty programs")

        print("[thinking with supermemory]")
        t1 = time.time()
        prefs = search_memories(user_id, " ".join(query_parts))
        add_supermemory(time.time() - t1)
        if prefs and "no memories" not in prefs.lower():
            parts.append(f"USER_PREFERENCES:\n{prefs}")

    has_prefs = bool(prefs and "no memories" not in prefs.lower())
    log_event(
        "prefetch",
        user_id=user_id or "",
        detected_intents=detected,
        has_prefs=has_prefs,
        latency_ms=(time.time() - t0) * 1000,
    )

    return {
        "commerce_context": "\n\n".join(parts) if parts else None,
        "active_intents": merged_intents,
    }


# ================= subgraph builder =================

_tool_node = ToolNode(TOOLS, handle_tool_errors=True)


def _pick_early_phrase(state: State) -> str | None:
    idx = _early_speak_idx_var.get()
    tool_names: list[str] = []
    for msg in reversed(state["messages"]):
        calls = getattr(msg, "tool_calls", None)
        if calls:
            tool_names = [
                (tc["name"] if isinstance(tc, dict) else tc.name) for tc in calls
            ]
            break

    for tool_name in _TOOL_SPEAK_PRIORITY:
        if tool_name in tool_names:
            phrases = _TOOL_SPEAK_MESSAGES[tool_name]
            if phrases:
                phrase = phrases[idx % len(phrases)]
                _early_speak_idx_var.set(idx + 1)
                return phrase
            return None  # tool explicitly silenced

    phrase = _FALLBACK_SPEAK_MESSAGES[idx % len(_FALLBACK_SPEAK_MESSAGES)]
    _early_speak_idx_var.set(idx + 1)
    return phrase


def _timed_tools(state: State):
    global _turn_tools_used

    if not _spoken_early_var.get():
        early_msg_text = _pick_early_phrase(state)
        _spoken_early_var.set(True)
        if early_msg_text:
            speak_early(early_msg_text)

    # Log each tool call being made
    for msg in reversed(state["messages"]):
        if calls := getattr(msg, "tool_calls", None):
            for tc in calls:
                name = tc["name"] if isinstance(tc, dict) else tc.name
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                log_event("tool_call", tool_name=name, args=str(args)[:300])
                _turn_tools_used.append(name)
            break

    _tools_running_var.set(True)
    t0 = time.time()
    try:
        result = _tool_node.invoke(state)
    finally:
        _tools_running_var.set(False)
    add_tool(time.time() - t0)

    # Log tool results.
    # ToolNode returns a dict normally; a list when any tool returns Command.
    log_messages: list = []
    if isinstance(result, dict):
        log_messages = result.get("messages", [])
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                log_messages.extend(item.get("messages", []))
            elif hasattr(item, "update") and isinstance(getattr(item, "update", None), dict):
                log_messages.extend(item.update.get("messages", []))

    for msg in log_messages:
        if isinstance(msg, ToolMessage):
            status = (
                "error"
                if (getattr(msg, "status", None) == "error"
                    or str(msg.content).lower().startswith("error"))
                else "ok"
            )
            log_event("tool_result", tool_name=msg.name or "", status=status,
                      content=str(msg.content)[:200])

    return result


def _build_subgraph():
    builder = StateGraph(State)
    builder.add_node("prefetch", _prefetch_node)
    builder.add_node("agent", _agent_node)
    builder.add_node("tools", _timed_tools)
    builder.add_node("repair", _repair_node)
    builder.add_conditional_edges(
        START, _prefetch_router,
        {"prefetch": "prefetch", "agent": "agent"},
    )
    builder.add_edge("prefetch", "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_conditional_edges(
        "tools", _check_tool_errors, {"repair": "repair", "agent": "agent"}
    )
    builder.add_edge("repair", "agent")
    return builder.compile()


# ================= init (location pre-load) =================

def _init_node(state: State) -> dict:
    """At conversation start, restore the user's last known location and flight from memory."""
    user_id = state.get("user_id")
    if not user_id or not MEMORY_ENABLED:
        return {}
    result: dict = {}
    if not state.get("current_location"):
        t0 = time.time()
        poi_id = get_last_location(user_id)
        add_supermemory(time.time() - t0)
        if poi_id:
            result["current_location"] = poi_id
    if not state.get("flight_number"):
        t0 = time.time()
        flight = get_last_flight(user_id)
        add_supermemory(time.time() - t0)
        if flight:
            result["flight_number"] = flight
    return result


# ================= main graph =================

def build_graph():
    main_subgraph = _build_subgraph()

    builder = StateGraph(State)
    builder.add_node("init", _init_node)
    builder.add_node("main", main_subgraph)

    builder.add_edge(START, "init")
    builder.add_edge("init", "main")
    builder.add_edge("main", END)

    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()
