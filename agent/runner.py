import time
import uuid

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from agent.graph import ITERATION_CAP, graph
from agent import timing


def run():
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": ITERATION_CAP,
    }

    print("OAK Airport Agent Ready")

    while True:
        user = input("\nYou: ")
        if user.lower() in ("exit", "quit"):
            break

        timing.reset()
        t0 = time.time()

        try:
            result = graph.invoke({"messages": [HumanMessage(content=user)]}, config)  # type: ignore
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
