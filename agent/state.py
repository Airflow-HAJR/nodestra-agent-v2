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
    user_location: Optional[dict]  # {lat, lng, accuracy, ts} — latest GPS fix streamed from the client, if any
    last_route: Optional[dict]  # structured route dict from the most recent get_route call (stops/segments/level_changes)
    active_segment_index: Optional[int]  # which floor segment of last_route is currently shown on the map
    route_id: Optional[str]  # uuid identifying last_route, so checkpoint events can be matched to the right route
