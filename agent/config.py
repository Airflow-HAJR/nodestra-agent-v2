import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://airflowbackendv2-production.up.railway.app"
DEFAULT_AIRPORT = "OAK"
HTTP_TIMEOUT = 20

AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
