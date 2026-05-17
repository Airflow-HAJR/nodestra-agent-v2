import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.llm import build_llm
from agent.memory import save_conversation
from agent.prompts import build_system_prompt
from agent.state import State
from agent.timing import add_llm, add_tool
from agent.tools import TOOLS

_EARLY_SPEAK_MESSAGES = [
    "Let me check that for you.",
    "One moment while I look that up.",
    "Checking the airport map now.",
    "I'll help you find that.",
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


# ================= subgraph builder =================

_tool_node = ToolNode(TOOLS, handle_tool_errors=True)


def _timed_tools(state: State):
    global _EARLY_SPEAK_IDX, tools_running

    if not _spoken_early_var.get():
        early_msg_text = _EARLY_SPEAK_MESSAGES[_EARLY_SPEAK_IDX % len(_EARLY_SPEAK_MESSAGES)]
        _EARLY_SPEAK_IDX += 1
        _spoken_early_var.set(True)
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


# ================= main graph =================

def _entry_router(state: State) -> Literal["clarify", "navigate"]:
    if not state.get("final_destination"):
        return "clarify"
    return "navigate"


def build_graph():
    clarify = _build_subgraph("clarify")
    navigate = _build_subgraph("navigate")

    builder = StateGraph(State)
    builder.add_node("clarify", clarify)
    builder.add_node("navigate", navigate)
    builder.add_node("closure", _closure_node)

    builder.add_conditional_edges(START, _entry_router, {"clarify": "clarify", "navigate": "navigate"})

    builder.add_edge("clarify", "closure")
    builder.add_edge("navigate", "closure")
    builder.add_edge("closure", END)

    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()
