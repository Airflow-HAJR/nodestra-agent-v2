"""
Open the interactive airport map in the browser.
Requires the API server to be running: uvicorn api.index:app --reload

Usage:
    python map.py           # opens AIRPORT_ID from .env
    python map.py LAX       # opens a specific airport
"""
import os
import sys
import webbrowser
from dotenv import load_dotenv

load_dotenv()

airport_id = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AIRPORT_ID", "")
if not airport_id:
    sys.exit("Set AIRPORT_ID in .env or pass it as an argument: python map.py SJC")

url = f"http://localhost:8000/map/{airport_id}"
print(f"Opening {url}")
webbrowser.open(url)
