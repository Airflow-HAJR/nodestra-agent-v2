from typing import Annotated, Any, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def take_latest(_existing: Any, new: Any) -> Any:
    """Last-write-wins reducer.

    A single agent turn can emit several tool calls that LangGraph runs together
    in one super-step (e.g. find_poi + navigate). Each tool returns a Command
    that writes state, and a plain channel rejects more than one write per step
    with InvalidUpdateError. These keys are all "current value" fields — the
    latest write is the one that should stick — so reconcile concurrent writes
    by keeping the last one applied instead of crashing the turn.
    """
    return new


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # Nav/flight state below is written by tools via Command. Several tools can
    # run in one super-step, so every key a tool may write carries take_latest
    # to tolerate concurrent writes (see take_latest above).
    current_location: Annotated[Optional[str], take_latest]
    final_destination: Annotated[Optional[str], take_latest]
    flight_number: Annotated[Optional[str], take_latest]
    last_error: Optional[str]
    should_end: Annotated[Optional[bool], take_latest]
    user_id: Optional[str]
    # True once the user has signed in: memories are written through to the
    # database and follow them to their next visit. False for guests, whose
    # facts live only in `user_memories` for the length of the conversation.
    persist_memory: Optional[bool]
    user_name: Optional[str]  # first name from the signed-in account, if any
    active_intents: Annotated[Optional[List[str]], take_latest]
    user_memories: Optional[List[dict]]  # ALL of this user's memories, loaded once at session start; refreshed on write
    # Whose memories `user_memories` actually holds. Compared against user_id
    # so signing in mid-conversation reloads the cache instead of leaving the
    # agent looking at the guest's.
    memories_user_id: Optional[str]
    relevant_memories: Optional[List[dict]]  # subset the recall_memory node judged relevant to the current message
    user_location: Optional[dict]  # {lat, lng, accuracy, ts} — latest GPS fix streamed from the client, if any
    language: Optional[str]  # ISO code the user picked in the UI; overrides mirroring whatever language they typed in
    last_route: Annotated[Optional[dict], take_latest]  # structured route dict from the most recent get_route call (stops/segments/level_changes)
    active_segment_index: Annotated[Optional[int], take_latest]  # which floor segment of last_route is currently shown on the map
    active_stop_index: Annotated[Optional[int], take_latest]  # index of the next unconfirmed checkpoint within that segment's stops
    route_id: Annotated[Optional[str], take_latest]  # uuid identifying last_route, so checkpoint events can be matched to the right route
    checkpoint_scripts: Annotated[Optional[List[str]], take_latest]  # pre-written "head toward next stop" messages, consumed one per advance_checkpoint
