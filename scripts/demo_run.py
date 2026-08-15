"""
Run the demo conversation headlessly and print everything the UI would show.

This is the rehearsal for the live demo: same graph, same tools, same memory,
but the map actions are printed instead of drawn and nothing is spoken. If a
beat doesn't land here it won't land on stage either.

    uv run python scripts/demo_run.py
    uv run python scripts/demo_run.py --user +15105550142
    uv run python scripts/demo_run.py --turn "im really hungry, whats nearby?"

Each turn prints: the tools the agent called, the map actions it emitted, and
the reply.
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402

from agent.graph import (  # noqa: E402
    ITERATION_CAP,
    bind_map_callback,
    get_turn_tools_used,
    graph,
    reset_turn_state,
    save_memory_in_background,
)
from agent.poi_rag import warm_index  # noqa: E402

DEMO_USER_ID = "+15105550142"

# Nothing is seeded: the agent starts out not knowing where the traveler is,
# says so, and asks — which is the honest state of things, since a phone's GPS
# can put you in the terminal but not at a gate. --start overrides it if you
# want to skip that beat.
DEMO_START_POI = ""

SCRIPT = [
    "hey im really hungry, what's nearby?",
    "im at terminal 1 security",
    "yeah let's do that one, take me there",
    # Confirming the checkpoint the agent actually named. Saying you're
    # somewhere further along ("I'm at gate 3") makes it advance one step and
    # then point you at where you already are — checkpoints are confirmed in
    # order, which is what the on-screen "Made it to __" button does for you.
    "made it to Lake Merritt Essentials",
]

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def _reply_of(result: dict) -> str:
    for m in reversed(result.get("messages", [])):
        if getattr(m, "content", None) and not getattr(m, "tool_calls", None):
            return m.content if isinstance(m.content, str) else str(m.content)
    return "(no reply)"


def run_turn(text: str, config: dict, state_seed: dict) -> dict:
    actions: list[dict] = []
    reset_turn_state()
    payload = {"messages": [HumanMessage(content=text)], **state_seed}

    print(f"\n{BOLD}USER ▸{RESET} {text}")
    t0 = time.time()
    with bind_map_callback(actions.append):
        result = graph.invoke(payload, config)  # type: ignore[arg-type]
    dt = time.time() - t0

    tools = get_turn_tools_used()
    print(f"{DIM}  tools:   {', '.join(tools) if tools else '(none)'}{RESET}")
    for a in actions:
        summary = a.get("type", "?")
        if a["type"] == "show_options":
            summary += ": " + ", ".join(f"{o['name']} ({o.get('note')})" for o in a["options"])
        elif a["type"] == "show_trajectory":
            seg = a["segments"][a["activeSegmentIndex"]]
            summary += f": {a['origin']['name']} → {a['destination']['name']} ({len(seg['stops'])} stops, {a.get('etaMinutes')} min)"
        elif a["type"] == "checkpoint_prompt":
            summary += f": {a['poiName']}"
        print(f"{DIM}  map:     {summary}{RESET}")
    print(f"{DIM}  latency: {dt:.1f}s{RESET}")
    print(f"{BOLD}LUNA ▸{RESET} {_reply_of(result)}")

    save_memory_in_background(config, result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=DEMO_USER_ID)
    ap.add_argument("--start", default=DEMO_START_POI,
                    help="POI id to pretend the traveler is already standing at (default: unknown, "
                         "so the agent asks)")
    ap.add_argument("--turn", action="append", help="override the scripted turns (repeatable)")
    ap.add_argument("--guest", action="store_true", help="run as a guest (no stored memories)")
    args = ap.parse_args()

    warm_index()

    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": ITERATION_CAP,
    }
    # Sent on every turn: LangGraph merges these into the checkpointed state, and
    # user_id/persist_memory are what the init node keys memory loading on.
    seed = {"user_id": args.user, "persist_memory": not args.guest}
    if args.start:
        seed["current_location"] = args.start

    for text in (args.turn or SCRIPT):
        run_turn(text, config, seed)
        seed = {"user_id": args.user, "persist_memory": not args.guest}

    # The memory writer is a background thread; give it a moment so its log
    # lines land before the process exits.
    time.sleep(3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
