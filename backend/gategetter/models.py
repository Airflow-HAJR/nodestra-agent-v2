"""
GateGetter data model definitions.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Flight:
    scheduled_time: Optional[str] = None
    actual_time: Optional[str] = None
    flight_number: Optional[str] = None
    destination_city: Optional[str] = None
    destination_iata: Optional[str] = None
    airline: Optional[str] = None
    terminal: Optional[str] = None
    gate: Optional[str] = None
    status: Optional[str] = None


@dataclass
class Change:
    time: str
    flight: str
    type: str
    field: str
    old_value: Optional[str]
    new_value: Optional[str]
    detail: str


@dataclass
class Notification:
    time: str
    phone: str
    flight: str
    msg: str


@dataclass
class AppState:
    tracked: list = field(default_factory=list)
    flights: list = field(default_factory=list)
    changes: list = field(default_factory=list)
    notifications: list = field(default_factory=list)
    last_poll: Optional[str] = None
    next_poll: Optional[str] = None
    poll_count: int = 0
    status: str = "starting"
