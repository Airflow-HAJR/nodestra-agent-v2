import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_AIRPORT = "OAK"

# Azure OpenAI
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
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

# Supermemory (persistent user personalization)
SUPERMEMORY_API_KEY = os.environ.get("SUPERMEMORY_API_KEY", "")

# Moss semantic search
MOSS_PROJECT_ID = os.environ.get("MOSS_PROJECT_ID", "")
MOSS_PROJECT_KEY = os.environ.get("MOSS_PROJECT_KEY", "")

# Cartesia TTS
CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY", "")
CARTESIA_VOICE_ID = os.environ.get("CARTESIA_VOICE_ID", "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4")
CARTESIA_MODEL_ID = os.environ.get("CARTESIA_MODEL_ID", "sonic-3.5")

# TTS provider: "elevenlabs" | "cartesia"
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "cartesia")

# Feature flags
MEMORY_ENABLED = os.environ.get("MEMORY_ENABLED", "false").lower() == "true"
