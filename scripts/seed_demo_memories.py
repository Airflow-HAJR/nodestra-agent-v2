"""
Seed a demo traveler's memory with what the agent "learned" on an earlier trip.

The memory system fills itself from conversation, which is the right behavior
and the wrong demo: the interesting moment is the agent recognizing a
preference it picked up *last time*, and there is no last time on a fresh
database. This writes that history directly.

    uv run python scripts/seed_demo_memories.py                  # seed
    uv run python scripts/seed_demo_memories.py --show           # print what's stored
    uv run python scripts/seed_demo_memories.py --reset          # wipe, then seed
    uv run python scripts/seed_demo_memories.py --user +15105551212

The default user id is DEMO_USER_ID below. Use that same id in the demo UI
(the "user id" box in webchat, or NODESTRA_DEMO_USER for the voice UI) or the
agent will meet a stranger.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.vector_memory import (  # noqa: E402
    add_memory,
    delete_all_memories,
    fetch_all_memories,
    is_duplicate,
)

DEMO_USER_ID = "+15105550142"

# What a previous trip through the airport would plausibly have taught the
# agent. metadata is what lets it say *why* it knows — see _format_memory in
# agent/prompts.py.
DEMO_MEMORIES: list[tuple[str, str, dict]] = [
    (
        "Only eats halal meat — asked whether three different restaurants were halal before picking one, "
        "and skipped the ones that could not confirm",
        "dietary",
        {"source": "previous trip", "airport": "OAK", "date": "2026-06-12"},
    ),
    (
        "Does not drink alcohol; prefers somewhere the bar is not the main attraction",
        "dietary",
        {"source": "previous trip", "airport": "OAK", "date": "2026-06-12"},
    ),
    (
        "Flies Southwest almost exclusively and books out of Terminal 1",
        "travel",
        {"source": "previous trip", "airport": "OAK", "date": "2026-06-12"},
    ),
    (
        "Travels with a partner and a 4-year-old, so prefers quick counter service over sit-down table service",
        "preference",
        {"source": "previous trip", "airport": "OAK", "date": "2026-03-28"},
    ),
    (
        "Carries a Chase Sapphire Reserve and has used it for lounge access before",
        "payment",
        {"source": "previous trip", "airport": "SFO", "date": "2026-03-28"},
    ),
]


def show(user_id: str) -> None:
    rows = fetch_all_memories(user_id)
    if not rows:
        print(f"(nothing stored for {user_id})")
        return
    print(f"{len(rows)} memories for {user_id}:")
    for r in rows:
        meta = r.get("metadata") or {}
        tag = f"  [{meta.get('source')}, {meta.get('airport')}, {meta.get('date')}]" if meta else ""
        print(f"  · ({r.get('category')}) {r.get('content')}{tag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=DEMO_USER_ID, help=f"user id to seed (default {DEMO_USER_ID})")
    ap.add_argument("--reset", action="store_true", help="delete every memory for this user first")
    ap.add_argument("--show", action="store_true", help="print stored memories and exit")
    args = ap.parse_args()

    if args.show:
        show(args.user)
        return 0

    if args.reset:
        n = delete_all_memories(args.user)
        print(f"deleted {n} existing memories for {args.user}")

    existing = fetch_all_memories(args.user)
    written = 0
    for content, category, metadata in DEMO_MEMORIES:
        if is_duplicate(existing, content):
            print(f"  = already there: {content[:60]}...")
            continue
        row = add_memory(args.user, content, category, metadata)
        if row is None:
            print(f"  ! FAILED to write: {content[:60]}...  (is Supabase configured?)", file=sys.stderr)
            continue
        existing.append(row)
        written += 1
        print(f"  + {category}: {content[:70]}...")

    print(f"\nseeded {written} new memories for {args.user}")
    show(args.user)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
