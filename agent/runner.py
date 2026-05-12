import uuid

from langchain_core.messages import HumanMessage

from agent.graph import graph


def run():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("OAK Airport Agent Ready")

    while True:
        user = input("\nYou: ")
        if user.lower() in ("exit", "quit"):
            break

        result = graph.invoke({"messages": [HumanMessage(content=user)]}, config)  # type: ignore

        for m in reversed(result["messages"]):
            if hasattr(m, "content") and m.content:
                print("\nAgent:", m.content)
                break
