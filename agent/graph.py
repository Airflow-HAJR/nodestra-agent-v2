import time

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.llm import build_llm
from agent.prompts import build_system_prompt
from agent.state import State
from agent.tools import TOOLS

_llm_with_tools = build_llm().bind_tools(TOOLS)


def agent(state: State):
    system = build_system_prompt(state)

    print("[AGENT] thinking...")
    t0 = time.time()

    response = _llm_with_tools.invoke([
        SystemMessage(content=system),
        *state["messages"],
    ])

    dt = time.time() - t0
    if response.tool_calls:
        names = [tc["name"] if isinstance(tc, dict) else tc.name for tc in response.tool_calls]
        print(f"[AGENT] {dt:.2f}s -> calling: {', '.join(names)}")
    else:
        content = response.content
        text = content if isinstance(content, str) else str(content)
        preview = text.replace("\n", " ")[:120]
        print(f"[AGENT] {dt:.2f}s -> reply: {preview}")

    return {"messages": [response]}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()
