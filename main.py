import time
import requests
from dotenv import load_dotenv
from typing import Annotated, Optional, List

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

load_dotenv()

BASE_URL = "https://airflowbackendv2-production.up.railway.app"

# ================= LLM =================

llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai"
)

# ================= STATE =================

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ================= TOOLS =================

class POIInput(BaseModel):
    q: str = Field(description="POI name like Gate 5, Escape Lounge")
    airport_id: str = Field(default="OAK")

class RouteInput(BaseModel):
    start: str = Field(description="start POI id (the 'id' field returned by find_poi, e.g. 'gate-CIU3')")
    end: str = Field(description="end POI id (the 'id' field returned by find_poi, e.g. 'lounge-MYAS')")
    airport_id: str = Field(default="OAK")

class GetNodesInput(BaseModel):
    airport_id: str = Field(default="OAK")
    floor: Optional[str] = Field(default=None, description="Optional floor filter, e.g. 'L1', 'L2'")

class ResolvePOIInput(BaseModel):
    name: str = Field(description="Ambiguous POI name (same name appears on multiple floors)")
    candidates: List[dict] = Field(description="List of POI candidates returned by find_poi")
    landmark: Optional[str] = Field(default=None, description="Nearby landmark hint from the user")
    floor: Optional[str] = Field(default=None, description="Floor hint if user provided one")
    airport_id: str = Field(default="OAK")

class FindNearestInput(BaseModel):
    source: str = Field(description="Source POI name or id to search from")
    poi_type: str = Field(description="Type of POI to find, e.g. 'restroom', 'gate', 'lounge'")
    airport_id: str = Field(default="OAK")

# -------- FIND POI --------

def _short(text: str, n: int = 300) -> str:
    return text if len(text) <= n else text[:n] + f"... ({len(text)} chars)"

@tool(args_schema=POIInput)
def find_poi(q: str, airport_id: str = "OAK"):
    """Fuzzy search for a single POI name. Returns a dict containing an 'id' field that should be passed to get_route."""

    print(f"[TOOL] find_poi q={q!r}")
    t0 = time.time()

    res = requests.get(
        f"{BASE_URL}/find-poi",
        params={"q": q, "airport_id": airport_id},
        timeout=20
    )

    print(f"  -> {res.status_code} in {time.time()-t0:.2f}s | {_short(res.text)}")
    return res.json()

# -------- ROUTE --------

@tool(args_schema=RouteInput)
def get_route(start: str, end: str, airport_id: str = "OAK"):
    """Get shortest path between two POIs. start and end must be POI ids returned by find_poi."""

    print(f"[TOOL] get_route start={start} end={end}")
    t0 = time.time()

    res = requests.get(
        f"{BASE_URL}/route",
        params={
            "start": start,
            "end": end,
            "airport_id": airport_id
        },
        timeout=20
    )

    print(f"  -> {res.status_code} in {time.time()-t0:.2f}s | {_short(res.text)}")
    return res.json()

# -------- GET NODES --------

@tool(args_schema=GetNodesInput)
def get_nodes(airport_id: str = "OAK", floor: Optional[str] = None):
    """List all POI nodes for an airport, optionally filtered by floor."""

    print(f"[TOOL] get_nodes floor={floor}")
    t0 = time.time()

    params = {"airport_id": airport_id}
    if floor:
        params["floor"] = floor

    res = requests.get(
        f"{BASE_URL}/get-nodes",
        params=params,
        timeout=20
    )

    print(f"  -> {res.status_code} in {time.time()-t0:.2f}s | {_short(res.text)}")
    return res.json()

# -------- RESOLVE POI --------

@tool(args_schema=ResolvePOIInput)
def resolve_poi(
    name: str,
    candidates: List[dict],
    landmark: Optional[str] = None,
    floor: Optional[str] = None,
    airport_id: str = "OAK",
):
    """Disambiguate POIs with the same name on different floors using a nearby landmark or floor hint."""

    print(f"[TOOL] resolve_poi name={name!r} landmark={landmark!r} floor={floor!r}")
    t0 = time.time()

    payload = {
        "name": name,
        "candidates": candidates,
        "airport_id": airport_id,
    }
    if landmark:
        payload["landmark"] = landmark
    if floor:
        payload["floor"] = floor

    res = requests.post(
        f"{BASE_URL}/resolve-poi",
        json=payload,
        timeout=20
    )

    print(f"  -> {res.status_code} in {time.time()-t0:.2f}s | {_short(res.text)}")
    return res.json()

# -------- FIND NEAREST --------

@tool(args_schema=FindNearestInput)
def find_nearest(source: str, poi_type: str, airport_id: str = "OAK"):
    """Find the closest POI of a given type from a source location."""

    print(f"[TOOL] find_nearest source={source!r} type={poi_type!r}")
    t0 = time.time()

    res = requests.post(
        f"{BASE_URL}/find-nearest",
        json={
            "source": source,
            "type": poi_type,
            "airport_id": airport_id,
        },
        timeout=20
    )

    print(f"  -> {res.status_code} in {time.time()-t0:.2f}s | {_short(res.text)}")
    return res.json()

tools = [find_poi, get_route, get_nodes, resolve_poi, find_nearest]

# one ToolNode per tool
find_poi_node = ToolNode([find_poi])
get_route_node = ToolNode([get_route])
get_nodes_node = ToolNode([get_nodes])
resolve_poi_node = ToolNode([resolve_poi])
find_nearest_node = ToolNode([find_nearest])

TOOL_TO_NODE = {
    "find_poi": "find_poi_node",
    "get_route": "get_route_node",
    "get_nodes": "get_nodes_node",
    "resolve_poi": "resolve_poi_node",
    "find_nearest": "find_nearest_node",
}

def route_tools(state: State):
    """Dispatch to the per-tool node based on the last AI message's tool call."""
    messages = state.get("messages", [])
    if not messages:
        return END

    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return END

    name = tool_calls[0].get("name") if isinstance(tool_calls[0], dict) else tool_calls[0].name
    return TOOL_TO_NODE.get(name, END)

llm_with_tools = llm.bind_tools(tools)

# ================= AGENT =================

def agent(state: State):

    system = """
You are an airport navigation assistant for OAK.

Tools:
- find_poi: fuzzy search for one location name → returns a dict with an 'id' field
- resolve_poi: if find_poi returns multiple candidates (same name, different floors), disambiguate with a nearby landmark or floor hint
- get_route: shortest path between two POIs — MUST be called with the 'id' values returned by find_poi, NOT the raw names
- find_nearest: closest POI of a given type from a source location
- get_nodes: list all POI nodes for the airport, optionally filtered by floor

ROUTING WORKFLOW (when the user asks how to get from A to B):
1. Call find_poi for A → take the 'id' from the result (e.g. 'gate-CIU3')
2. Call find_poi for B → take the 'id' from the result (e.g. 'lounge-MYAS')
3. If either find_poi returned multiple candidates, call resolve_poi first
4. Call get_route(start=<id from step 1>, end=<id from step 2>)
5. Summarize the route to the user

Never guess routes or POI data. Never pass raw names to get_route — always pass ids.
"""

    print("[AGENT] thinking...")
    t0 = time.time()

    response = llm_with_tools.invoke([
        SystemMessage(content=system),
        *state["messages"]
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

# ================= GRAPH =================

builder = StateGraph(State)

builder.add_node("agent", agent)
builder.add_node("find_poi_node", find_poi_node)
builder.add_node("get_route_node", get_route_node)
builder.add_node("get_nodes_node", get_nodes_node)
builder.add_node("resolve_poi_node", resolve_poi_node)
builder.add_node("find_nearest_node", find_nearest_node)

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    route_tools,
    {
        "find_poi_node": "find_poi_node",
        "get_route_node": "get_route_node",
        "get_nodes_node": "get_nodes_node",
        "resolve_poi_node": "resolve_poi_node",
        "find_nearest_node": "find_nearest_node",
        END: END,
    },
)

builder.add_edge("find_poi_node", "agent")
builder.add_edge("get_route_node", "agent")
builder.add_edge("get_nodes_node", "agent")
builder.add_edge("resolve_poi_node", "agent")
builder.add_edge("find_nearest_node", "agent")

graph = builder.compile()

# ================= RUNNER =================

def run():

    state: State = {"messages": []}

    print("OAK Airport Agent Ready")

    while True:

        user = input("\nYou: ")
        if user.lower() in ["exit", "quit"]:
            break

        state["messages"].append(HumanMessage(content=user))

        state = graph.invoke(state) #type: ignore

        # print final response
        for m in reversed(state["messages"]):
            if hasattr(m, "content") and m.content:
                print("\nAgent:", m.content)
                break


if __name__ == "__main__":
    run()