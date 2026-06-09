import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.llm import build_llm
from agent.memory import get_last_flight, get_last_location, get_user_profile
from agent.prompts import build_system_prompt
from agent.state import State
from agent.timing import add_llm, add_tool
import logging

logger = logging.getLogger(__name__)

_turn_tools_used: list[str] = []


def get_turn_tools_used() -> list[str]:
    return list(_turn_tools_used)
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
    "get_nodes": [
        "Pulling up the airport map.",
        "Loading the map for you.",
    ],
    "set_nav_state": [],  # silent — just a state update
}
_TOOL_SPEAK_PRIORITY = ["get_route", "find_nearest", "find_poi", "resolve_poi", "get_nodes"]
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
        print("[thinking]")


@contextmanager
def bind_speak_early_callback(callback: Callable[[str], None] | None) -> Iterator[None]:
    token = _speak_early_callback_var.set(callback)
    try:
        yield
    finally:
        _speak_early_callback_var.reset(token)


def reset_turn_state() -> None:
    global _turn_tools_used
    _spoken_early_var.set(False)
    _turn_tools_used = []


ITERATION_CAP = 30

_llm_with_tools = build_llm().bind_tools(TOOLS)


# ================= phase agent =================

def _make_agent(phase: str):
    def agent(state: State):
        system = build_system_prompt(state, phase=phase)
        print("[thinking]")
        t0 = time.time()

        response = _llm_with_tools.invoke([
            SystemMessage(content=system),
            *state["messages"],
        ])

        dt = time.time() - t0
        add_llm(dt)

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
    print("[thinking]")
    return {"last_error": err[:300]}


# ================= commerce prefetch =================


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
    global tools_running, _turn_tools_used

    if not _spoken_early_var.get():
        early_msg_text = _pick_early_phrase(state)
        _spoken_early_var.set(True)
        if early_msg_text:
            speak_early(early_msg_text)

    # Collect tool calls from last AI message
    batch: list[dict] = []
    for msg in reversed(state["messages"]):
        calls = getattr(msg, "tool_calls", None)
        if calls:
            batch = [
                {"name": tc["name"] if isinstance(tc, dict) else tc.name,
                 "args": tc["args"] if isinstance(tc, dict) else tc.args}
                for tc in calls
            ]
            break

    for tc in batch:
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in tc["args"].items()) if tc["args"] else ""
        print(f"  → {tc['name']}({args_str})")
        _turn_tools_used.append(tc["name"])

    tools_running = True
    t0 = time.time()
    try:
        result = _tool_node.invoke(state)
    finally:
        tools_running = False
    elapsed = time.time() - t0
    add_tool(elapsed, len(batch) or 1)

    # Print tool results
    for msg in (result.get("messages") or []):
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            truncated = content[:200] + "..." if len(content) > 200 else content
            print(f"  ← {truncated}")

    return result


def _build_subgraph(phase: str):
    builder = StateGraph(State)
    builder.add_node("agent", _make_agent(phase))
    builder.add_node("tools", _timed_tools)
    builder.add_node("repair", _repair_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_conditional_edges(
        "tools", _check_tool_errors, {"repair": "repair", "agent": "agent"}
    )
    builder.add_edge("repair", "agent")
    return builder.compile()


# ================= closure (heuristic, no LLM) =================

_DECLINE_PHRASES = ("no thanks", "that's all", "i'm good", "bye", "goodbye", "no i'm set", "nope", "nah", "i'm set", "thank you", "thanks bye")


def _closure_node(state: State) -> dict:
    """Check if the user's latest message is a decline. If so, set should_end."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            lower = (msg.content if isinstance(msg.content, str) else str(msg.content)).lower().strip()
            if any(lower == p or lower.startswith(p) for p in _DECLINE_PHRASES):
                return {"should_end": True}
            break
    return {}


# ================= init (location pre-load) =================

def _init_node(state: State) -> dict:
    user_id = state.get("user_id")
    if not user_id:
        return {}
    result: dict = {}
    if not state.get("current_location"):
        poi_id = get_last_location(user_id)
        if poi_id:
            result["current_location"] = poi_id
    if not state.get("flight_number"):
        flight = get_last_flight(user_id)
        if flight:
            result["flight_number"] = flight
    if not state.get("user_profile"):
        profile = get_user_profile(user_id)
        if profile:
            result["user_profile"] = profile
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
