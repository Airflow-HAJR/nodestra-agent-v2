import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel as _PydanticBase
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.config import MEMORY_ENABLED
from agent.llm import build_llm
from agent.prompts import build_system_prompt
from agent.state import State
from agent.timing import add_llm, add_memory, add_tool
from agent.vector_memory import (
    add_memory as vm_add_memory,
    fetch_all_memories,
    is_duplicate,
)
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
    "track_flight_changes": [
        "Setting up flight tracking for you.",
        "Registering your flight alert.",
    ],
}
_TOOL_SPEAK_PRIORITY = ["get_route", "find_nearest", "find_poi", "resolve_poi", "get_nodes", "track_flight_changes"]
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
        # detect_intent and set_nav_state repopulate them for this turn.
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


# ================= intent detection =================

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
        _intent_classifier = build_llm(fast=True).with_structured_output(_IntentResult, method="json_mode")
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


def _detect_intent_node(state: State) -> dict:
    """Classify the user's intent so PHASE_FOCUS renders the right instructions.
    Sets active_intents; does not touch memory."""
    detected = _detect_intents(state["messages"])
    existing = list(state.get("active_intents") or [])
    merged_intents = list({*existing, *detected})
    print(f"[intents detected: {detected or 'none'}]")
    return {"active_intents": merged_intents}


# ================= memory recall =================

_MEMORY_SELECT_SYSTEM = """\
You help an airport assistant decide which of the things it remembers about a user \
are useful for the user's LATEST message.

You are given the user's stored memories and their latest message. Return the subset \
of those memories (copied verbatim) that would help you respond well.

Guidance:
- If the user is asking what you know / remember about them, return ALL memories.
- Include a memory if it should shape your answer (e.g. they ask about food and you \
know a dietary restriction; they ask for directions and you know a mobility need).
- If nothing is relevant, return an empty list.

Respond with JSON only: {"relevant": ["<memory text>", ...]}\
"""


class _RelevantMemories(_PydanticBase):
    relevant: list[str]


_memory_selector = None


def _get_memory_selector():
    global _memory_selector
    if _memory_selector is None:
        _memory_selector = build_llm(fast=True).with_structured_output(
            _RelevantMemories, method="json_mode"
        )
    return _memory_selector


def _memory_recall_node(state: State) -> dict:
    """Look at everything we remember about the user (loaded once at session start)
    and pick the subset relevant to their latest message. The chosen memories are
    surfaced prominently in the system prompt so the agent actually uses them."""
    if not MEMORY_ENABLED:
        return {}
    memories = state.get("user_memories") or []
    if not memories:
        return {"relevant_memories": []}

    last_user = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user = m.content if isinstance(m.content, str) else str(m.content)
            break
    if not last_user:
        return {"relevant_memories": []}

    listing = "\n".join(f"- {m['content']}" for m in memories)
    try:
        t0 = time.time()
        result = _get_memory_selector().invoke([
            SystemMessage(content=_MEMORY_SELECT_SYSTEM),
            HumanMessage(content=f"Stored memories:\n{listing}\n\nUser's latest message: \"{last_user}\""),
        ])
        add_llm(time.time() - t0)
    except Exception as e:
        print(f"[memory recall failed: {e}]")
        return {"relevant_memories": memories}  # safe fallback: give the agent everything

    chosen = [c.strip().lower() for c in (result.relevant or [])]
    relevant = [
        m for m in memories
        if any(m["content"].strip().lower() == c or m["content"].strip().lower() in c or c in m["content"].strip().lower()
               for c in chosen)
    ]
    print(f"[memory recall: {[m['content'] for m in relevant] or 'nothing relevant'}]")
    return {"relevant_memories": relevant}


# ================= memory extraction (save) =================

_MEMORY_EXTRACT_SYSTEM = """\
You decide whether the user's latest message contains DURABLE personal facts \
worth remembering for future conversations with an airport assistant.

Remember things like: dietary restrictions or allergies (halal, vegan, nut allergy), \
accessibility needs (uses a wheelchair, avoids stairs), preferred airline, loyalty or \
payment cards, home city, who they travel with, and lasting preferences.

Do NOT remember: one-off requests ("take me to gate 5"), questions, today-only details \
(a specific flight number for this trip), small talk, or acknowledgements.

Break the message into ATOMIC facts — ONE preference per fact, each with its own \
category. Never combine unrelated facts (e.g. a diet and a mobility need) into one entry.

Respond with JSON only:
{"remember": true/false, "facts": [{"content": "<concise third-person fact>", "category": "dietary|payment|accessibility|travel|preference|personal|other"}]}

If nothing is worth remembering: {"remember": false, "facts": []}\
"""


class _MemoryFact(_PydanticBase):
    content: str
    category: str


class _MemoryExtraction(_PydanticBase):
    remember: bool
    facts: list[_MemoryFact]


_memory_extractor = None


def _get_memory_extractor():
    global _memory_extractor
    if _memory_extractor is None:
        _memory_extractor = build_llm(fast=True).with_structured_output(
            _MemoryExtraction, method="json_mode"
        )
    return _memory_extractor


def _memory_extract_node(state: State) -> dict:
    """After the agent's final answer, decide if the user's last message held a
    durable fact. If so, persist it and append it to the session cache so it's
    recallable for the rest of this call without re-querying the DB."""
    if not MEMORY_ENABLED:
        return {}
    user_id = state.get("user_id")
    if not user_id:
        return {}

    last_user = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user = m.content if isinstance(m.content, str) else str(m.content)
            break
    if not last_user:
        return {}

    try:
        result = _get_memory_extractor().invoke([
            SystemMessage(content=_MEMORY_EXTRACT_SYSTEM),
            HumanMessage(content=last_user),
        ])
    except Exception as e:
        print(f"[memory extract failed: {e}]")
        return {}

    if not result or not result.remember or not result.facts:
        return {}

    cached = list(state.get("user_memories") or [])
    saved = 0
    for fact in result.facts:
        content = (fact.content or "").strip()
        if not content:
            continue
        if is_duplicate(cached, content):
            print(f"[memory: skipped duplicate — {content!r}]")
            continue
        t0 = time.time()
        row = vm_add_memory(user_id, content, fact.category or "other")
        add_memory(time.time() - t0)
        if row:
            cached.append(row)
            saved += 1
            print(f"[memory: saved ({row['category']}) — {row['content']!r}]")

    return {"user_memories": cached} if saved else {}


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


def _after_agent_router(state: State) -> Literal["tools", "save_memory"]:
    """Tool-call turns go to the tools node; a final answer goes to save_memory
    (which decides what, if anything, to remember) before ending."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "save_memory"


def _build_subgraph():
    builder = StateGraph(State)
    builder.add_node("detect_intent", _detect_intent_node)
    builder.add_node("recall_memory", _memory_recall_node)
    builder.add_node("agent", _agent_node)
    builder.add_node("tools", _timed_tools)
    builder.add_node("repair", _repair_node)
    builder.add_node("save_memory", _memory_extract_node)

    builder.add_edge(START, "detect_intent")
    builder.add_edge("detect_intent", "recall_memory")
    builder.add_edge("recall_memory", "agent")
    builder.add_conditional_edges(
        "agent", _after_agent_router,
        {"tools": "tools", "save_memory": "save_memory"},
    )
    builder.add_edge("save_memory", END)
    builder.add_conditional_edges(
        "tools", _check_tool_errors, {"repair": "repair", "agent": "agent"}
    )
    builder.add_edge("repair", "agent")
    return builder.compile()


# ================= init (location pre-load) =================

def _init_node(state: State) -> dict:
    """At conversation start, bulk-load the user's semantic memories once."""
    user_id = state.get("user_id")
    if not user_id or not MEMORY_ENABLED:
        return {}

    # Bulk-load the user's memories once per session.
    # Recall for the rest of the session runs against this cache — no per-turn DB hit.
    if state.get("user_memories") is not None:
        return {}

    t0 = time.time()
    memories = fetch_all_memories(user_id)
    add_memory(time.time() - t0)
    print(f"[memory: loaded {len(memories)} stored memories for session]")
    return {"user_memories": memories}


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
