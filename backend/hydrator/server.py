"""
Flight hydrator: fetches live flights from GateGetter and upserts them
as vector documents into Supabase pgvector for agent search.

Toggle on/off with: FLIGHT_HYDRATION_ENABLED=true  (default: false)

Other env vars:
  GATEGETTER_URL               default: http://localhost:8081
  HYDRATION_AIRPORT            default: OAK
  HYDRATION_INTERVAL_MINUTES   default: 5
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AzureOpenAI
from supabase import create_client, Client

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [hydrator] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
HYDRATION_ENABLED     = os.environ.get("FLIGHT_HYDRATION_ENABLED", "false").lower() == "true"
GATEGETTER_URL        = os.environ.get("GATEGETTER_URL", "http://localhost:8081")  # mirrors agent/config.py
HYDRATION_AIRPORT     = os.environ.get("HYDRATION_AIRPORT", "OAK")
HYDRATION_INTERVAL    = int(os.environ.get("HYDRATION_INTERVAL_MINUTES", "5")) * 60
SUPABASE_URL          = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY", "")
OAI_API_KEY           = os.environ.get("AZURE_OPENAI_API_KEY", "")
OAI_ENDPOINT          = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
OAI_API_VERSION       = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
EMBEDDING_DEPLOYMENT  = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

_DATA_DIR = Path(__file__).parent.parent.parent / "data"

# ── Lazy clients ──────────────────────────────────────────────────────────────
_openai: AzureOpenAI | None = None
_supabase: Client | None = None


def _get_openai() -> AzureOpenAI:
    global _openai
    if _openai is None:
        _openai = AzureOpenAI(
            api_key=OAI_API_KEY,
            azure_endpoint=OAI_ENDPOINT,
            api_version=OAI_API_VERSION,
        )
    return _openai


def _get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase


def _embed(text: str) -> list[float]:
    resp = _get_openai().embeddings.create(input=text, model=EMBEDDING_DEPLOYMENT)
    return resp.data[0].embedding


# ── Text converters ───────────────────────────────────────────────────────────
def _flight_to_text(f: dict) -> str:
    dest_city = f.get("destination_city", "")
    dest_iata = f.get("destination_iata", "")
    destination = f"{dest_city} ({dest_iata})" if dest_iata else dest_city
    departs = f.get("actual_time") or f.get("scheduled_time", "TBD")
    return (
        f"Flight {f.get('flight_number', '?')} ({f.get('airline', '?')}) to {destination}. "
        f"Departs {departs}. "
        f"{f.get('terminal', '')} {f.get('gate', '')}. Status: {f.get('status', 'Unknown')}."
    ).strip()


def _store_to_text(s: dict) -> str:
    parts = [
        f"{s['name']} — {s['category']}. Located in {s.get('terminal', '?')}, {s.get('gate_area', '')}.",
        f"Hours: {s.get('hours', 'unknown')}.",
        s.get("description", ""),
    ]
    if s.get("dietary"):
        parts.append(f"Dietary info: {s['dietary']}")
    if s.get("perks"):
        parts.append(f"Card perks & loyalty: {s['perks']}")
    parts.append(f"Payment: {s.get('accepts', 'major cards')}.")
    return " ".join(parts)


# ── Upsert ────────────────────────────────────────────────────────────────────
def _upsert(rows: list[dict]) -> None:
    if not rows:
        return
    _get_supabase().table("vector_documents").upsert(rows, on_conflict="id").execute()


def seed_stores(airport_id: str) -> None:
    path = _DATA_DIR / "oak_stores.json"
    if not path.exists():
        log.warning("Store data not found at %s — skipping store seed", path)
        return
    stores: list[dict] = json.loads(path.read_text())
    rows = []
    for i, s in enumerate(stores):
        content = _store_to_text(s)
        rows.append({
            "id": f"store-{airport_id}-{i}",
            "collection": "stores",
            "airport_id": airport_id,
            "content": content,
            "metadata": s,
            "embedding": _embed(content),
        })
    _upsert(rows)
    log.info("Seeded %d stores for %s", len(rows), airport_id)


def hydrate_flights(airport_id: str) -> int:
    url = f"{GATEGETTER_URL}/api/data?airport={airport_id}"
    log.info("Fetching flights from %s", url)
    try:
        resp = httpx.get(url, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        log.error("GateGetter request failed: %s", e)
        return 0

    flights: list[dict] = resp.json().get("flights", [])
    if not flights:
        log.info("No flights returned for %s", airport_id)
        return 0

    rows = []
    for i, f in enumerate(flights):
        fn = f.get("flight_number") or f"unk-{i}"
        content = _flight_to_text(f)
        rows.append({
            "id": f"flight-{airport_id}-{fn}",
            "collection": "flights",
            "airport_id": airport_id,
            "content": content,
            "metadata": f,
            "embedding": _embed(content),
        })
    _upsert(rows)
    return len(rows)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    if not HYDRATION_ENABLED:
        log.info("Hydrator is off (FLIGHT_HYDRATION_ENABLED is not 'true') — exiting.")
        return

    missing = [k for k, v in {
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_SERVICE_KEY / SUPABASE_KEY": SUPABASE_SERVICE_KEY,
        "AZURE_OPENAI_API_KEY": OAI_API_KEY,
        "AZURE_OPENAI_ENDPOINT": OAI_ENDPOINT,
    }.items() if not v]
    if missing:
        log.error("Missing required env vars: %s", ", ".join(missing))
        return

    log.info("Starting — airport=%s, interval=%ds, gategetter=%s",
             HYDRATION_AIRPORT, HYDRATION_INTERVAL, GATEGETTER_URL)

    log.info("Seeding stores...")
    try:
        seed_stores(HYDRATION_AIRPORT)
    except Exception as e:
        log.error("Store seed failed: %s", e)

    while True:
        try:
            n = hydrate_flights(HYDRATION_AIRPORT)
            log.info("Hydrated %d flights", n)
        except Exception as e:
            log.error("Flight hydration error: %s", e)
        log.info("Next hydration in %ds", HYDRATION_INTERVAL)
        time.sleep(HYDRATION_INTERVAL)


if __name__ == "__main__":
    main()
