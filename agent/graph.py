import time
from concurrent.futures import ThreadPoolExecutor
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
from agent.map_tools import _emit_checkpoint_prompt, _emit_trajectory, _lookup_gps, _map_callback_var

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
    "show_map_trajectory": [],    # silent — map update, no filler needed
    "request_checkpoint_confirmation": [],  # silent — the agent speaks prompt_text itself
    "advance_checkpoint": [],     # silent — map update, no filler needed
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
        # active_intents is reset each turn; the navigate/set_nav_state tools
        # repopulate it, and _derive_phases also infers "navigate" from
        # final_destination, so the navigation phase survives without a
        # separate intent-classification LLM call.
        response = _llm_with_tools.invoke([
            SystemMessage(content=system),
            *state["messages"],
        ])

    dt = time.time() - t0
    add_llm(dt)

    return {"messages": [response], "last_error": None, "active_intents": []}


# ================= post-tool routing =================

def _check_tool_errors(state: State) -> Literal["agent", "checkpoint_reply"]:
    """After tools run, decide where to go next.

    - advance_checkpoint with a pre-written script queued -> checkpoint_reply
      (zero LLM: the "head toward the next stop" line was written by navigate).
    - everything else -> back to the agent, which reads the tool result (errors
      included, since the error text is in the ToolMessage) and writes the reply.

    There is no separate repair/format tier anymore: the agent model is capable
    enough to both recover from a tool error and verbalize a tool result itself,
    so paying for extra LLM nodes to do those jobs was pure latency.
    """
    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            break
        if msg.name == "advance_checkpoint" and state.get("checkpoint_scripts"):
            return "checkpoint_reply"
    return "agent"


def _checkpoint_reply_node(state: State) -> dict:
    """Return the next pre-scripted checkpoint message without calling the LLM.

    Pops the first entry from checkpoint_scripts (written by navigate at route-compute
    time) and returns it directly as the agent's reply. This eliminates the second
    main-LLM call that would otherwise turn an advance_checkpoint result into a
    'now head toward X' message.
    """
    scripts = list(state.get("checkpoint_scripts") or [])
    if not scripts:
        return {}
    print(f"[checkpoint fast-path: '{scripts[0][:60]}...']")
    return {
        "messages": [AIMessage(content=scripts[0])],
        "checkpoint_scripts": scripts[1:],
    }


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
            _MemoryExtraction, method="function_calling"
        )
    return _memory_extractor


# Memory saving happens ENTIRELY off the critical path. The reply is already
# back in the user's hands; a background worker then decides whether the last
# message held a durable fact, persists it, and patches it into the thread's
# checkpoint so the next turn's prompt includes it. Zero LLM calls on the turn
# itself — that's the whole point.
_MEMORY_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mem-save")


def _extract_and_persist(config: dict, state: dict) -> None:
    """Background: extract durable facts from the last user message and save them.

    Runs after the turn has already returned, so nothing here is timed or blocks
    the user. When a fact is found it's written to the DB (signed-in users only)
    and merged into the thread checkpoint via graph.update_state, so the next
    turn's system prompt carries it. Guests get the checkpoint update but no DB
    write — the fact lives for the session and no longer.
    """
    if not MEMORY_ENABLED:
        return
    user_id = state.get("user_id")
    if not user_id:
        return
    persist = bool(state.get("persist_memory"))

    last_user = None
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            last_user = m.content if isinstance(m.content, str) else str(m.content)
            break
    if not last_user:
        return

    # No keyword pre-filter — the extraction model itself decides whether the
    # message holds anything durable. It runs in the background after the reply,
    # so an extra call costs the user nothing, and a regex gate would silently
    # miss real facts (typos, unusual phrasings) it never thought to list.
    try:
        result = _get_memory_extractor().invoke([
            SystemMessage(content=_MEMORY_EXTRACT_SYSTEM),
            HumanMessage(content=last_user),
        ])
    except Exception as e:
        print(f"[memory extract failed: {e}]")
        return
    if not result or not result.remember or not result.facts:
        return

    cached = list(state.get("user_memories") or [])
    saved = 0
    for fact in result.facts:
        content = (fact.content or "").strip()
        if not content or is_duplicate(cached, content):
            continue
        category = fact.category or "other"
        row = None
        if persist:
            try:
                row = vm_add_memory(user_id, content, category)
            except Exception as e:
                print(f"[memory: DB write failed — {e}]")
        cached.append(row or {"id": None, "content": content, "category": category, "metadata": {}})
        saved += 1
        print(f"[memory: saved ({category}, {'db' if row else 'session'}) — {content!r}]")

    if saved:
        try:
            graph.update_state(config, {"user_memories": cached})
        except Exception as e:
            print(f"[memory: checkpoint update failed — {e}]")


def save_memory_in_background(config: dict, result: dict) -> None:
    """Fire-and-forget memory extraction for a just-completed turn. Call this
    after graph.invoke() returns — it never blocks and adds no LLM call to the
    turn the user is waiting on."""
    _MEMORY_POOL.submit(_extract_and_persist, config, dict(result))


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


def _extract_tool_state_update(result, key: str):
    """Pull a state key out of a ToolNode result — a dict for a plain tool,
    or a list of dict/Command entries when any tool in the batch returns
    Command (get_route among them, for last_route/route_id)."""
    if isinstance(result, dict):
        return result.get(key)
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and key in item:
                return item[key]
            upd = getattr(item, "update", None)
            if isinstance(upd, dict) and key in upd:
                return upd[key]
    return None


def _auto_emit_map_from_tools(state: State, result, log_messages: list) -> None:
    """Show the map automatically based on get_route results, so the map
    reflects the trajectory the agent just planned whether or not it
    separately remembers to call show_map_directions/show_map_trajectory. A
    bare POI lookup (find_poi/find_nearest/resolve_poi) isn't a trajectory
    and isn't worth popping the map open for on its own — only routes
    trigger this."""
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
            # Prefer the floor-by-floor trajectory — with every named POI
            # along the way shown as a numbered dot, not just the two
            # endpoints — for every route, single-level or multi-level. Only
            # single-level routes ever collapse to one segment; that's still
            # strictly better than a flat two-point line since it shows the
            # stops in between. Also auto-surface the first checkpoint's
            # confirm button, so it appears whether or not the agent
            # separately calls show_map_trajectory this turn.
            raw = _extract_tool_state_update(result, "last_route")
            route_id = _extract_tool_state_update(result, "route_id") or ""
            if raw and _emit_trajectory(raw, route_id, 0, 1):
                _emit_checkpoint_prompt(raw, route_id, 0, 1)
                continue

            # GPS lookup failed for the trajectory (e.g. intermediate POIs
            # unresolvable) — fall back to a flat directions line, re-deriving
            # GPS from the tool call's own args since get_route's ToolMessage
            # content is prose.
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

    _auto_emit_map_from_tools(state, result, log_messages)

    return result


def _after_agent_router(state: State) -> Literal["tools", "end"]:
    """Tool-call turns go to the tools node; a final answer ends the turn.
    Memory saving is NOT a node — it runs in the background after invoke()."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


def _build_subgraph():
    """Minimal agent loop — the agent (Luna) does everything: tool selection,
    error recovery, and writing the reply. The only other node is the zero-LLM
    checkpoint reply. Memory extraction happens off-graph, in the background,
    after the turn returns (see save_memory_in_background).

        START -> agent -> END
                   \\-> tools -> agent -> END
                          \\-> checkpoint_reply -> END
    """
    builder = StateGraph(State)
    builder.add_node("agent", _agent_node)
    builder.add_node("tools", _timed_tools)
    builder.add_node("checkpoint_reply", _checkpoint_reply_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", _after_agent_router,
        {"tools": "tools", "end": END},
    )
    builder.add_conditional_edges(
        "tools", _check_tool_errors,
        {"agent": "agent", "checkpoint_reply": "checkpoint_reply"},
    )
    builder.add_edge("checkpoint_reply", END)
    return builder.compile()


# ================= init (location pre-load) =================

def _init_node(state: State) -> dict:
    """Load the memories that belong to whoever is currently on the line.

    Runs on every turn but does real work only when the identity behind the
    conversation changes — which is once at session start for most sessions,
    and a second time for anyone who signs in partway through.

    Signing in mid-conversation is the interesting case. Guests still
    accumulate facts (they're just never written to the database), so at the
    moment an account appears we have two sets: what this conversation learned,
    and what the account already knew. Both are kept — the session's facts are
    flushed to the account, so nothing the user just said is lost by the act of
    signing in.
    """
    user_id = state.get("user_id")
    if not user_id or not MEMORY_ENABLED:
        return {}

    # Same person as last turn — the cache built below is still theirs.
    if state.get("memories_user_id") == user_id and state.get("user_memories") is not None:
        return {}

    persist = bool(state.get("persist_memory"))
    if not persist:
        # A guest: nothing to load, and nothing they say will outlive the
        # conversation. Start them an empty cache so this doesn't re-run.
        return {"user_memories": [], "memories_user_id": user_id}

    t0 = time.time()
    memories = fetch_all_memories(user_id)

    # Carry over anything learned before the user signed in.
    carried = [
        m for m in (state.get("user_memories") or [])
        if not is_duplicate(memories, m.get("content") or "")
    ]
    for m in carried:
        row = vm_add_memory(user_id, m.get("content") or "", m.get("category") or "other")
        if row:
            memories.append(row)
    if carried:
        print(f"[memory: carried {len(carried)} session facts into the account]")

    add_memory(time.time() - t0)
    print(f"[memory: loaded {len(memories)} stored memories for session]")
    return {"user_memories": memories, "memories_user_id": user_id}


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
