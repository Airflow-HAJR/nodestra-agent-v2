import time
import uuid

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from agent.graph import ITERATION_CAP, graph
from agent.memory import save_conversation
from agent import timing


def run(user_id: str | None = None):
    if not user_id:
        user_id = input("User ID (phone number or leave blank for guest): ").strip() or None

    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": ITERATION_CAP,
    }

    # Pass user_id into initial state so tools can access it via InjectedState
    initial_state: dict = {}
    if user_id:
        initial_state["user_id"] = user_id

    print("OAK Airport Agent Ready")

    result = None
    while True:
        user = input("\nYou: ")
        if user.lower() in ("exit", "quit"):
            if user_id and result:
                save_conversation(user_id, result["messages"])
            break

        timing.reset()
        t0 = time.time()

        payload = {**initial_state, "messages": [HumanMessage(content=user)]}
        # user_id only needs to be in the first invocation; LangGraph checkpointer persists state
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

        for m in reversed(result["messages"]):
            if hasattr(m, "content") and m.content:
                print("\nAgent:", m.content)
                break

        print(timing.summary(total))

        if result.get("should_end"):
            print("\n[conversation ended]")
            break
