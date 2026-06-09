import time
import uuid

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from agent.analytics import finish_call, hash_user_id, insert_turn, start_call, upsert_user_memory
from agent.config import DEFAULT_AIRPORT
from agent.graph import ITERATION_CAP, graph
from agent.memory import save_conversation
from agent.graph import get_turn_tools_used
from agent.summarizer import summarize
from agent import timing


def run(user_id: str | None = None):
    if not user_id:
        user_id = input("User ID (phone number or leave blank for guest): ").strip() or None

    thread_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": ITERATION_CAP,
    }

    initial_state: dict = {}
    if user_id:
        initial_state["user_id"] = user_id

    timing.call_reset()
    call_started_at = time.time()
    uid_hash = hash_user_id(user_id) if user_id else None
    start_call(
        call_id=session_id,
        user_id_hash=uid_hash,
        started_at=call_started_at,
    )

    print("OAK Airport Agent Ready")

    result = None
    while True:
        user = input("\nYou: ")

        if user.strip() == "END":
            state_snap: dict = result or {}
            messages = state_snap.get("messages", []) if result else []
            call_stats = timing.call_snapshot()
            summary_data = summarize(messages, state_snap)
            finish_call(
                call_id=session_id,
                duration_s=time.time() - call_started_at,
                turn_count=call_stats["turn_count"],
                timing=call_stats,
                flight_number=state_snap.get("flight_number") if result else None,
                topics=summary_data["topics"],
                resolved=summary_data["resolved"],
                summary=summary_data["summary"],
            )
            if uid_hash and result:
                upsert_user_memory(
                    uid_hash, DEFAULT_AIRPORT,
                    last_flight=state_snap.get("flight_number"),
                    last_location=state_snap.get("current_location"),
                )
            print("\n[session ended — analytics written]")
            break

        if user.lower() in ("exit", "quit"):
            if user_id and result:
                save_conversation(user_id, result["messages"])
            break

        timing.reset()
        t0 = time.time()
        turn_number = timing._call["turn_count"] + 1

        payload = {**initial_state, "messages": [HumanMessage(content=user)]}
        initial_state = {}

        try:
            result = graph.invoke(payload, config)  # type: ignore
        except GraphRecursionError:
            print(
                "\nAgent: I'm getting tangled up trying to work that out. "
                "Could you re-state what you need?"
            )
            continue

        total = time.time() - t0
        turn_stats = timing.turn_snapshot(total)
        tools_used = get_turn_tools_used()

        insert_turn(
            call_id=session_id,
            turn_number=turn_number,
            turn_stats=turn_stats,
            tools_used=tools_used,
        )

        for m in reversed(result["messages"]):
            if hasattr(m, "content") and m.content:
                print("\nAgent:", m.content)
                break

        print(timing.summary(total))

        if result.get("should_end"):
            print("\n[conversation ended]")
            break
