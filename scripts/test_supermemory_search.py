#!/usr/bin/env python3
import argparse
import json
from typing import Any

from supermemory import Supermemory

from agent.config import SUPERMEMORY_API_KEY


def _value(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_dict(obj: Any):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return str(obj)


def main():
    parser = argparse.ArgumentParser(description="Test Supermemory search without running the agent.")
    parser.add_argument("--user-id", required=True, help="Supermemory container tag / user id")
    parser.add_argument("--query", required=True, help="Search query, e.g. 'food preferences'")
    args = parser.parse_args()

    client = Supermemory(api_key=SUPERMEMORY_API_KEY)

    print(f"[TEST] profile container_tag={args.user_id}")
    profile_resp = client.profile(container_tag=args.user_id)
    print(json.dumps(_to_dict(profile_resp), indent=2, default=str))

    print(f"[TEST] search.memories q={args.query!r} container_tag={args.user_id} limit=5 threshold=0.6")
    search_resp = client.search.memories(
        q=args.query,
        container_tag=args.user_id,
        search_mode="memories",
        limit=5,
        threshold=0.6,
    )

    results = _value(search_resp, "results", []) or []
    print(f"[TEST] results_count={len(results)}")

    for i, r in enumerate(results, start=1):
        memory = _value(r, "memory", "")
        score = _value(r, "score", None)
        print(f"{i}. score={score} memory={memory}")

    print("\n[TEST] full_search_response")
    print(json.dumps(_to_dict(search_resp), indent=2, default=str))


if __name__ == "__main__":
    main()
