import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel as _PydanticBase
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import MEMORY_ENABLED
from agent.llm import build_llm
from agent.memory import get_last_flight, get_last_location_with_name, get_session_metadata, search_memories
from agent.prompts import build_system_prompt
from agent.state import State
from agent.timing import add_llm, add_memory, add_tool
from agent.tools import TOOLS
from agent.logger import log_event
from agent.map_tools import _lookup_gps, _map_callback_var

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
    "set_nav_state": [],          # silent — just a state update
    "end_call": [],               # silent — just a state update
    "show_map_destination": [],   # silent — map update, no filler needed
    "show_map_directions": [],    # silent — map update, no filler needed
    "show_map_route": [],         # silent — map update, no filler needed
    "clear_map": [],              # silent — map update, no filler needed
    "search_flight_info": [
        "Checking the departure board.",
        "Looking up that flight.",
    ],
    "search_store_info": [
        "Looking up the shops and restaurants.",
        "Checking what's available.",
    ],
    "track_flight_changes": [
        "Setting up flight tracking for you.",
        "Registering your flight alert.",
    ],
}
_TOOL_SPEAK_PRIORITY = ["get_route", "find_nearest", "find_poi", "resolve_poi", "recall_user_memories", "get_nodes", "search_flight_info", "search_store_info", "track_flight_changes"]
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
_tool_status_callback_var: ContextVar[Callable[[str], None] | None] = ContextVar(
    "tool_status_callback",
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
def bind_tool_status_callback(callback: Callable[[str], None] | None) -> Iterator[None]:
    """Bind a callback invoked once per turn with the name of the tool the
    agent is about to run (e.g. "get_route") — lets callers surface a
    tool-specific status update (e.g. to a UI) without affecting speech."""
    token = _tool_status_callback_var.set(callback)
    try:
        yield
    finally:
        _tool_status_callback_var.reset(token)


@contextmanager
def bind_sentence_callback(callback: Callable[[str | None], None] | None) -> Iterator[None]:
    token = _sentence_callback_var.set(callback)
    try:
        yield
    finally:
        _sentence_callback_var.reset(token)


@contextmanager
def bind_map_callback(callback: Callable[[dict], None] | None) -> Iterator[None]:
    from agent.map_tools import _map_callback_var
    token = _map_callback_var.set(callback)
    try:
        yield
    finally:
        _map_callback_var.reset(token)


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

# Intents that trigger a memory prefetch when detected
_PREFETCH_INTENTS = {"navigate", "food_drinks", "shopping", "lounge_access", "payment"}

_INTENT_SYSTEM = """\
You are an intent classifier for an airport voice assistant.

Classify the intent of the LAST USER MESSAGE only. Use prior conversation only to resolve ambiguous references \
(e.g. "it", "there", "that place", "yeah let's do it") — do not re-classify intents that were already handled \
in earlier turns.

Intents:
- navigate: user wants to go somewhere or get directions — includes confirmations like "yeah let's do it" \
or "take me there" when the prior assistant turn offered navigation
- food_drinks: user is actively requesting food or drink recommendations (not already in progress)
- shopping: user wants to buy something or find a specific shop
- lounge_access: user is asking about airport lounges
- payment: user is asking about payment methods, cards, or rewards
- flight_info: user is asking about flight status, gate, boarding, or delays
- accessibility: user has mobility needs or is asking about wheelchair/elevator access
- ground_transport: user is asking about taxis, rideshare, BART, or airport shuttles
- baggage: user is asking about luggage or bag claim
- wellness: user is looking for a spa, quiet room, chapel, or relaxation space
- charging_connectivity: user needs device charging, wifi, or a power outlet
- family_services: user needs family restrooms, play area, or stroller info

Return {"intents": []} for simple acknowledgements with no new request ("okay thanks", "got it", "sounds good").

Respond with JSON only.\
"""

class _IntentResult(_PydanticBase):
    intents: list[str]

_intent_classifier = None


def _get_intent_classifier():
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = build_llm().with_structured_output(_IntentResult, method="json_mode")
    return _intent_classifier


def _detect_intents(messages: list) -> list[str]:
    """Use the LLM to classify the user's intent given recent conversation context."""
    # Build a compact context string from the last 4 messages so short replies like
    # "yeah let's do it" are understood relative to what was just discussed.
    context_lines: list[str] = []
    for msg in messages[-4:]:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            context_lines.append(f"User: {content}")
        elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            context_lines.append(f"Assistant: {content[:200]}")
    context = "\n".join(context_lines)
    try:
        result = _get_intent_classifier().invoke([
            SystemMessage(content=_INTENT_SYSTEM),
            HumanMessage(content=context),
        ])
        return result.intents if result else []
    except Exception as e:
        print(f"[intent detection failed: {e}]")
        return []


def _prefetch_router(state: State) -> Literal["prefetch", "agent"]:
    """Always route to prefetch — LLM detection inside prefetch decides what (if anything) to fetch."""
    return "prefetch"


def _prefetch_node(state: State) -> dict:
    """Pre-fetch personalization data for intents where it meaningfully affects the response.

    Queries user memory for navigate, food_drinks, shopping, lounge_access, and payment intents.
    Sets specific detected intent names in active_intents so PHASE_FOCUS renders the right instructions.
    """
    # Detect intents from recent conversation context (not just the last message)
    detected = _detect_intents(state["messages"])

    # Merge with any intents already in state (e.g. navigate set by set_nav_state)
    existing = list(state.get("active_intents") or [])
    merged_intents = list({*existing, *detected})

    print(f"[intents detected: {detected or 'none'}]")

    if not MEMORY_ENABLED:
        print("[memory disabled — skipping preference prefetch]")
        return {"commerce_context": None, "active_intents": merged_intents}

    t0 = time.time()
    parts: list[str] = []
    prefs: str | None = None
    user_id = state.get("user_id")

    active_prefetch = [i for i in detected if i in _PREFETCH_INTENTS]
    needs_navigate = "navigate" in active_prefetch
    needs_food = "food_drinks" in active_prefetch
    needs_lounge = "lounge_access" in active_prefetch
    # Co-occurrence rules: food/shopping always pull payment; navigate always pulls accessibility
    needs_payment = "payment" in active_prefetch or "shopping" in active_prefetch or needs_food
    needs_accessibility = needs_navigate  # navigate always checks mobility/accessibility prefs

    effective_intents = list(active_prefetch)
    if needs_payment and "payment" not in effective_intents:
        effective_intents.append("payment")
    if needs_accessibility and "accessibility" not in effective_intents:
        effective_intents.append("accessibility")
    print(f"[prefetch intents: {effective_intents}]")

    # Map each effective intent to a short, clean memory query term
    _INTENT_QUERY: dict[str, str] = {
        "navigate":     "route mobility accessibility",
        "food_drinks":  "food diet cuisine",
        "shopping":     "shopping preference",
        "lounge_access":"lounge membership",
        "payment":      "payment card loyalty",
        "accessibility":"mobility wheelchair elevator",
    }

    if user_id and effective_intents:
        query_terms = " ".join(
            _INTENT_QUERY[i] for i in effective_intents if i in _INTENT_QUERY
        )
        query = query_terms
        print(f"[memory query: {query!r}]")
        t1 = time.time()
        prefs = search_memories(user_id, query)
        add_memory(time.time() - t1)
        print(f"[memory result: {prefs!r}]")
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


# ================= preference acknowledgment =================

def _pref_ack_router(state: State) -> Literal["pref_ack", "agent"]:
    """Route to pref_ack only on the first turn preferences are found; then let agent handle it."""
    if state.get("pref_acked"):
        return "agent"
    commerce = state.get("commerce_context") or ""
    if "USER_PREFERENCES:" in commerce:
        return "pref_ack"
    return "agent"


def _pref_ack_node(state: State) -> dict:
    """Use the LLM to generate a fun, natural preference-acknowledgment and ask the user to choose.

    Skips the main agent this turn. On the next turn, the agent sees pref_acked=True and
    routes straight to agent which calls search_store_info with the user's stated choice.
    """
    commerce = state.get("commerce_context") or ""
    # Parse "key: value" lines out of the USER_PREFERENCES block
    pref_lines: list[str] = []
    in_block = False
    for line in commerce.splitlines():
        if line.strip() == "USER_PREFERENCES:":
            in_block = True
            continue
        if in_block and line.strip():
            pref_lines.append(line.strip())

    if not pref_lines:
        return {}

    last_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_msg = m.content if isinstance(m.content, str) else str(m.content)
            break

    # Pass the full key:value list so the LLM sees food AND payment preferences
    prefs_block = "\n".join(pref_lines)

    prompt = (
        f"The user just said: \"{last_msg}\"\n\n"
        f"Their stored preferences from past visits:\n{prefs_block}\n\n"
        "In 1-2 casual spoken sentences: tell them you're pulling up their past preferences, "
        "reference the most relevant ones for what they asked (cover both food AND payment if both are present), "
        "then ask if they want to go with one of those or try something different. "
        "Be natural and warm. No lists, no markdown."
    )

    t0 = time.time()
    response = build_llm().invoke([SystemMessage(content=prompt)])
    add_llm(time.time() - t0)

    msg = response.content if isinstance(response.content, str) else str(response.content)
    speak_early(msg)
    return {"messages": [AIMessage(content=msg)], "pref_acked": True}


# ================= subgraph builder =================

_tool_node = ToolNode(TOOLS, handle_tool_errors=True)


def _pick_active_tool_name(state: State) -> str | None:
    """The name of the highest-priority tool call the agent is about to run
    (the last message's tool_calls), or None if there isn't one."""
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
            return tool_name
    return tool_names[0] if tool_names else None


def _pick_early_phrase(state: State, tool_name: str | None = None) -> str | None:
    idx = _early_speak_idx_var.get()
    if tool_name is None:
        tool_name = _pick_active_tool_name(state)

    if tool_name and tool_name in _TOOL_SPEAK_MESSAGES:
        phrases = _TOOL_SPEAK_MESSAGES[tool_name]
        if phrases:
            phrase = phrases[idx % len(phrases)]
            _early_speak_idx_var.set(idx + 1)
            return phrase
        return None  # tool explicitly silenced

    phrase = _FALLBACK_SPEAK_MESSAGES[idx % len(_FALLBACK_SPEAK_MESSAGES)]
    _early_speak_idx_var.set(idx + 1)
    return phrase


def _auto_emit_map_from_tools(state: State, log_messages: list) -> None:
    """Show the map automatically based on get_route results, so the map
    reflects the trajectory the agent just planned whether or not it
    separately remembers to call show_map_directions. A bare POI lookup
    (find_poi/find_nearest/resolve_poi) isn't a trajectory and isn't worth
    popping the map open for on its own — only routes trigger this."""
    cb = _map_callback_var.get()
    if cb is None or not log_messages:
        return

    calls_by_id: dict[str, tuple[str, dict]] = {}
    for msg in reversed(state["messages"]):
        if calls := getattr(msg, "tool_calls", None):
            for tc in calls:
                tc_id = tc["id"] if isinstance(tc, dict) else tc.id
                name = tc["name"] if isinstance(tc, dict) else tc.name
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                calls_by_id[tc_id] = (name, args)
            break

    for msg in log_messages:
        if not isinstance(msg, ToolMessage):
            continue
        call = calls_by_id.get(msg.tool_call_id)
        if not call:
            continue
        name, args = call

        if name == "get_route":
            # get_route's return value is a speech-formatted string, not
            # structured JSON — but its call args are already the POI ids we
            # need, so look those up directly instead of parsing the string.
            start_gps = _lookup_gps(args.get("start", ""))
            end_gps = _lookup_gps(args.get("end", ""))
            if start_gps and end_gps:
                cb({
                    "type": "show_directions",
                    "destination": {"name": end_gps["name"], "lat": end_gps["lat"], "lng": end_gps["lng"]},
                    "origin": {"lat": start_gps["lat"], "lng": start_gps["lng"]},
                })


def _timed_tools(state: State):
    global _turn_tools_used

    if not _spoken_early_var.get():
        tool_name = _pick_active_tool_name(state)

        status_cb = _tool_status_callback_var.get()
        if status_cb and tool_name:
            status_cb(tool_name)

        early_msg_text = _pick_early_phrase(state, tool_name)
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

    _auto_emit_map_from_tools(state, log_messages)

    return result


def _build_subgraph():
    builder = StateGraph(State)
    builder.add_node("prefetch", _prefetch_node)
    builder.add_node("agent", _agent_node)
    builder.add_node("tools", _timed_tools)
    builder.add_node("repair", _repair_node)
    builder.add_node("pref_ack", _pref_ack_node)
    builder.add_conditional_edges(
        START, _prefetch_router,
        {"prefetch": "prefetch", "agent": "agent"},
    )
    builder.add_conditional_edges(
        "prefetch", _pref_ack_router,
        {"pref_ack": "pref_ack", "agent": "agent"},
    )
    builder.add_edge("pref_ack", END)
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
    # Only run on the very first turn — user_profile persists in checkpoint after that
    if state.get("user_profile"):
        return {}
    result: dict = {}

    t0 = time.time()
    meta = get_session_metadata(user_id)
    add_memory(time.time() - t0)

    if meta:
        result["user_profile"] = {
            "visit_count": meta.get("visit_count"),
            "last_flight": meta.get("last_flight"),
            "last_location": meta.get("last_location"),
            "last_location_name": meta.get("last_location_name"),
            "last_seen": meta.get("last_seen"),
        }
        visit = meta.get("visit_count") or 0
        last_loc = meta.get("last_location_name") or "unknown"
        last_flight = meta.get("last_flight") or "unknown"
        print(f"[session: visit #{visit}, last location: {last_loc}, last flight: {last_flight}]")

    if not state.get("current_location") and not state.get("suggested_location"):
        poi_id = meta.get("last_location") if meta else None
        poi_name = meta.get("last_location_name") if meta else None
        if not poi_id:
            t0 = time.time()
            poi_id, poi_name = get_last_location_with_name(user_id)
            add_memory(time.time() - t0)
        if poi_id:
            result["suggested_location"] = {"id": poi_id, "name": poi_name or poi_id}

    if not state.get("flight_number"):
        flight = meta.get("last_flight") if meta else None
        if not flight:
            t0 = time.time()
            flight = get_last_flight(user_id)
            add_memory(time.time() - t0)
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
