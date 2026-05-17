#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv


TOP_K = 7


async def run(query: str) -> int:
    load_dotenv(".env")

    moss_project_id = os.environ.get("MOSS_PROJECT_ID", "")
    moss_project_key = os.environ.get("MOSS_PROJECT_KEY", "")

    print(f"MOSS_PROJECT_ID set: {bool(moss_project_id)}")
    print(f"MOSS_PROJECT_KEY set: {bool(moss_project_key)}")

    if not moss_project_id or not moss_project_key:
        print("\nMoss is not configured. Add both env vars to .env:")
        print("MOSS_PROJECT_ID=...")
        print("MOSS_PROJECT_KEY=...")
        return 2

    from moss import MossClient, QueryOptions

    client = MossClient(moss_project_id, moss_project_key)
    indexes = ["oakland-pois", "oakland-flights"]
    ready: list[str] = []

    print("\nLoading indexes...")
    for idx in indexes:
        try:
            await client.load_index(idx)
            ready.append(idx)
            print(f"  load_index({idx}): OK")
        except Exception as e:
            print(f"  load_index({idx}): FAIL -> {type(e).__name__}: {e}")

    if not ready:
        print("\nNo indexes are loadable. Check project credentials and index names.")
        return 3

    print(f"\nQuery: {query!r} (top_k={TOP_K})")
    for idx in ready:
        try:
            res = await client.query(idx, query, QueryOptions(top_k=TOP_K))
            docs = getattr(res, "docs", []) or []
            print(f"\nIndex {idx}: {len(docs)} docs")
            for i, doc in enumerate(docs, start=1):
                text = (getattr(doc, "text", "") or "").replace("\n", " ")
                score = getattr(doc, "score", None)
                print(f"{i}. score={score} text={text[:180]}")
        except Exception as e:
            print(f"\nIndex {idx}: query failed -> {type(e).__name__}: {e}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Moss search debug tool.")
    parser.add_argument("--query", required=True, help="Search query")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.query)))


if __name__ == "__main__":
    main()
