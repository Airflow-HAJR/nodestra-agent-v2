import time
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from agent.llm import build_llm
from agent.prompts import build_system_prompt
from agent.state import State
from agent.tools import TOOLS

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

def _build_subgraph(phase: str):
    builder = StateGraph(State)
    builder.add_node("agent", _make_agent(phase))
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    builder.add_node("repair", _repair_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_conditional_edges(
        "tools", _check_tool_errors, {"repair": "repair", "agent": "agent"}
    )
    builder.add_edge("repair", "agent")
    return builder.compile()


# ================= closure detection =================

class _ClosureDecision(BaseModel):
    decision: Literal["ongoing", "task_complete", "user_declined"]


_closure_llm = build_llm().with_structured_output(_ClosureDecision)


_CLOSURE_PROMPT = """\
You classify whether an airport navigation conversation is wrapping up.

Pick exactly one label:
- "user_declined": the user's latest message declines further help (e.g. "no thanks", "that's all", "I'm good", "bye", "no I'm set").
- "task_complete": the agent's latest reply sounds like the immediate task just finished (e.g. "you've arrived at Gate 5") AND the agent has not yet asked whether the user needs anything else.
- "ongoing": anything else — active routing, mid-question, gathering info, tool follow-ups.
"""


def _closure_node(state: State):
    last_user = ""
    last_ai: AIMessage | None = None
    for msg in reversed(state["messages"]):
        if last_ai is None and isinstance(msg, AIMessage) and msg.content:
            last_ai = msg
        elif not last_user and isinstance(msg, HumanMessage):
            last_user = msg.content if isinstance(msg.content, str) else str(msg.content)
        if last_ai is not None and last_user:
            break

    if last_ai is None:
        return {}

    last_ai_text = last_ai.content if isinstance(last_ai.content, str) else str(last_ai.content)

    try:
        raw = _closure_llm.invoke([
            SystemMessage(content=_CLOSURE_PROMPT),
            HumanMessage(content=f"User: {last_user}\nAgent: {last_ai_text}"),
        ])
    except Exception as e:
        print(f"[CLOSURE] classifier failed: {e}")
        return {}

    decision = raw.get("decision") if isinstance(raw, dict) else getattr(raw, "decision", None)
    print(f"[CLOSURE] decision={decision}")

    if decision == "user_declined":
        return {"should_end": True}

    if decision == "task_complete":
        lowered = last_ai_text.lower()
        if "anything else" not in lowered and "anything more" not in lowered:
            updated = AIMessage(
                content=last_ai_text.rstrip() + " Anything else I can help with?",
                id=last_ai.id,
            )
            return {"messages": [updated]}

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
    builder.add_conditional_edges(
        START, _entry_router, {"clarify": "clarify", "navigate": "navigate"}
    )
    builder.add_edge("clarify", "closure")
    builder.add_edge("navigate", "closure")
    builder.add_edge("closure", END)
    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()
