from typing import Annotated, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_location: Optional[str]
    final_destination: Optional[str]
    flight_number: Optional[str]
    last_error: Optional[str]
    should_end: Optional[bool]
    user_id: Optional[str]
    active_intents: Optional[List[str]]
    user_memories: Optional[List[dict]]  # ALL of this user's memories, loaded once at session start; refreshed on write
    relevant_memories: Optional[List[dict]]  # subset the recall_memory node judged relevant to the current message
