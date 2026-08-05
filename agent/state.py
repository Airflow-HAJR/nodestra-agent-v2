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
    # True once the user has signed in: memories are written through to the
    # database and follow them to their next visit. False for guests, whose
    # facts live only in `user_memories` for the length of the conversation.
    persist_memory: Optional[bool]
    user_name: Optional[str]  # first name from the signed-in account, if any
    active_intents: Optional[List[str]]
    user_memories: Optional[List[dict]]  # ALL of this user's memories, loaded once at session start; refreshed on write
    # Whose memories `user_memories` actually holds. Compared against user_id
    # so signing in mid-conversation reloads the cache instead of leaving the
    # agent looking at the guest's.
    memories_user_id: Optional[str]
    relevant_memories: Optional[List[dict]]  # subset the recall_memory node judged relevant to the current message
    user_location: Optional[dict]  # {lat, lng, accuracy, ts} — latest GPS fix streamed from the client, if any
    language: Optional[str]  # ISO code the user picked in the UI; overrides mirroring whatever language they typed in
    last_route: Optional[dict]  # structured route dict from the most recent get_route call (stops/segments/level_changes)
    active_segment_index: Optional[int]  # which floor segment of last_route is currently shown on the map
    active_stop_index: Optional[int]  # index of the next unconfirmed checkpoint within that segment's stops
    route_id: Optional[str]  # uuid identifying last_route, so checkpoint events can be matched to the right route
    checkpoint_scripts: Optional[List[str]]  # pre-written "head toward next stop" messages, consumed one per advance_checkpoint
