import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_AIRPORT = "OAK"

# LLM provider: "openai" | "azure"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()

# OpenAI (default provider). Two tiers:
#   MAIN — capable model for the agent (the part that matters)
#   FAST — small, fast model for helper nodes (intent, recall, save, summary)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL_MAIN = os.environ.get("OPENAI_MODEL_MAIN", "gpt-4o")
OPENAI_MODEL_FAST = os.environ.get("OPENAI_MODEL_FAST", "gpt-4.1-nano")

# Reasoning models (the gpt-5.6-* family: luna/terra/sol) reject function tools
# on /v1/chat/completions unless reasoning is disabled — the agent binds TOOLS,
# so this must be 'none' for them. For the voice agent we want reasoning off
# anyway (latency). Leave EMPTY for non-reasoning models like gpt-4o, which
# don't accept the parameter at all and would 400 if it were sent.
OPENAI_MAIN_REASONING_EFFORT = os.environ.get("OPENAI_MAIN_REASONING_EFFORT", "")

# Groq — powers the FAST tier (helper nodes: intent, recall, save, format) when a
# key is set. Runs Llama on LPU hardware at ~750 tok/s, so a one-sentence reply
# comes back in ~100-250ms vs a hosted GPT's 400-700ms. OpenAI-API-compatible, so
# it's just a base_url swap. Falls back to OPENAI_MODEL_FAST when unset.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL_FAST = os.environ.get("GROQ_MODEL_FAST", "llama-3.1-8b-instant")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# The model name shown in latency breakdowns for the fast tier — whichever is live.
FAST_LLM_LABEL = GROQ_MODEL_FAST if GROQ_API_KEY else OPENAI_MODEL_FAST

# Azure OpenAI (active provider when LLM_PROVIDER=azure). One deployment serves
# both LLM tiers; for a gpt-5.6-* reasoning deployment (Luna) set
# OPENAI_MAIN_REASONING_EFFORT=none so it can bind tools.
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Supabase (map data)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Twilio
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")

# ElevenLabs
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # default: George

# Deepgram (transcription)
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

# Public base URL of this server (needed for Twilio to fetch <Play> audio).
# Falls back to Railway's auto-injected domain when SERVER_BASE_URL is not set explicitly.
_railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN") or os.environ.get("RAILWAY_STATIC_URL")
SERVER_BASE_URL = (
    os.environ.get("SERVER_BASE_URL")
    or (f"https://{_railway_domain}" if _railway_domain else "http://localhost:8000")
)

# Moss semantic search
MOSS_PROJECT_ID = os.environ.get("MOSS_PROJECT_ID", "")
MOSS_PROJECT_KEY = os.environ.get("MOSS_PROJECT_KEY", "")

# Cartesia TTS
CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY", "")
CARTESIA_VOICE_ID = os.environ.get("CARTESIA_VOICE_ID", "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4")
CARTESIA_MODEL_ID = os.environ.get("CARTESIA_MODEL_ID", "sonic-3.5")
CARTESIA_EMOTION = os.environ.get("CARTESIA_EMOTION", "enthusiastic")

# TTS provider: "elevenlabs" | "cartesia"
#
# ElevenLabs by default because its model is multilingual on a single voice:
# one voice id speaks all twelve UI languages when language_code pins the
# language, so the traveler hears the same guide whichever one they switch to.
# It is also the only provider whose voice exposes an expressiveness control
# (Cartesia has speed, but its `emotion` field accepts any string at all
# without validation, so there is nothing there to build a slider on).
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "elevenlabs")

# ── Voice tuning ──
# What the traveler can move in the settings sheet, and the bounds the server
# clamps to no matter what a client sends.
#
# The speed range is the intersection of the two providers' own limits —
# ElevenLabs accepts 0.7–1.2 and rejects anything outside it, Cartesia accepts
# 0.6–1.5 — so a single slider means the same thing on both and can never
# produce a 400 by drifting out of range.
VOICE_SPEED_MIN = 0.7
VOICE_SPEED_MAX = 1.2
VOICE_SPEED_DEFAULT = 1.0

# Expressiveness is the slider; ElevenLabs' knob is `stability`, which runs the
# other way (high stability = flat and even). The two are mirrored at the point
# of use so the UI can say "lively" where the API says 0.
VOICE_EXPRESSIVENESS_DEFAULT = 0.5

# ── Per-language TTS voices ───────────────────────────────────────────────────
# A Spanish reply read by an American voice sounds like an American reading
# Spanish, so each language gets a voice native to it. These are Cartesia's
# stock voices (all verified against sonic-3.5) — female "guide / support"
# personas throughout so the agent keeps one character across languages and
# only the accent changes.
CARTESIA_VOICE_IDS: dict[str, str] = {
    "en": "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",  # Skylar - Friendly Guide
    "es": "db74bd0c-9ea6-4d08-b78e-c3c0a54dfd2d",  # Ines - Route Guide
    "fr": "3b7d569e-01fc-45ef-b74b-29460956c691",  # Josette - Frontline Helper
    "de": "0b66a153-548f-4f2c-b734-09a13b0bd163",  # Lorelei - Helpful Guide
    "it": "0e21713a-5e9a-428a-bed4-90d410b87f13",  # Alessandra - Melodic Guide
    "pt": "d4b44b9a-82bc-4b65-b456-763fce4c52f9",  # Beatriz - Support Guide
    "ru": "25b7aaa6-1670-42dc-b791-419322400803",  # Daria - Decisive Dispatcher
    "zh": "7a5d4663-88ae-47b7-808e-8f9b9ee4127b",  # Hua - Sunny Support
    "ja": "d0ff6870-dd30-420d-8568-d756d806ea62",  # Hinata - Graceful Guide
    "ko": "ce9ca2b6-2bed-4452-99bb-052e1ec0b534",  # Seoyun - Warm Guide
    "ar": "731ace69-ee17-41bc-8c6f-665c9f1db95c",  # Fatima - Graceful Guide
    "hi": "bec003e2-3cb3-429c-8468-206a393c67ad",  # Parvati - Friendly Supporter
}

# ElevenLabs equivalents. Only the languages this account actually has a
# native-accent voice for are listed — the resolver falls through to Cartesia
# for the rest rather than having an American voice read Japanese.
ELEVENLABS_VOICE_IDS: dict[str, str] = {
    "en": ELEVENLABS_VOICE_ID,
    "es": "nfyTTmgO0f6GV9CKrMWL",  # Valeria — Latin American female
    "hi": "dVTC43Yewy5fAIcmsISI",  # Anvi — Hindi female
}


def _voice_overrides(prefix: str, table: dict[str, str]) -> dict[str, str]:
    """Let any single language be re-pointed from the environment, e.g.
    CARTESIA_VOICE_ID_ES=... , without touching the table above."""
    merged = dict(table)
    for code in set(table) | set(CARTESIA_VOICE_IDS):
        override = os.environ.get(f"{prefix}_{code.upper()}")
        if override:
            merged[code] = override.strip()
    return merged


CARTESIA_VOICE_IDS = _voice_overrides("CARTESIA_VOICE_ID", CARTESIA_VOICE_IDS)
ELEVENLABS_VOICE_IDS = _voice_overrides("ELEVENLABS_VOICE_ID", ELEVENLABS_VOICE_IDS)

# GateGetter service URL (flight tracking)
GATEGETTER_URL = os.environ.get("GATEGETTER_URL", "http://localhost:8081")

# Feature flags
MEMORY_ENABLED = os.environ.get("MEMORY_ENABLED", "true").lower() == "true"
FLIGHT_HYDRATION_ENABLED = os.environ.get("FLIGHT_HYDRATION_ENABLED", "false").lower() == "true"
FLIGHT_TRACKER_ENABLED = os.environ.get("FLIGHT_TRACKER_ENABLED", "false").lower() == "true"

# Comma-separated list of origins allowed to hit the HTTP/WS API, e.g.
# "https://agent.nodestra.com". Defaults to "*" (any origin) for local dev —
# set explicitly in production.
_allowed_origins_raw = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]
