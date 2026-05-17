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

# AgentPhone
AGENTPHONE_API_KEY = os.environ.get("AGENTPHONE_API_KEY")
AGENTPHONE_WEBHOOK_SECRET = os.environ.get("AGENTPHONE_WEBHOOK_SECRET")

# Supermemory (persistent user personalization)
SUPERMEMORY_API_KEY = os.environ.get("SUPERMEMORY_API_KEY", "")
