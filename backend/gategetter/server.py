"""
GateGetter Flight Tracker — API Server

Headless API service. Serves JSON endpoints — scraping happens on demand
when GET /api/data is called. No background threads.

Usage:
    python3 server.py
    SCRAPER_PORT=8081 python3 server.py
"""

import json
import os
import signal
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from airports import build_url, list_supported, AIRPORT_SLUGS
from state import (
    airports_state, state_lock, shutdown_event,
    ensure_airport, get_airport_state, remove_airport,
    add_tracked, remove_tracked, add_subscriber,
)
from scraper import scrape_once

PORT = int(os.environ.get("SCRAPER_PORT", "8081"))


class ReuseHTTPServer(HTTPServer):
    allow_reuse_address = True


def _parse_airport(qs: dict) -> str | None:
    """Extract airport code from query string, default to first active airport."""
    vals = qs.get("airport", [])
    if vals:
        return vals[0].upper()
    with state_lock:
        codes = list(airports_state.keys())
    return codes[0] if codes else None


class Handler(BaseHTTPRequestHandler):

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = parse_qs(parsed.query)

        if path == "/api/data":
            airport = _parse_airport(qs)
            if not airport:
                self._json_response(400, {"error": "no airport specified — pass ?airport=OAK"})
                return
            try:
                scrape_once(airport)
            except Exception as e:
                self._json_response(500, {"error": str(e)})
                return
            data = get_airport_state(airport)
            data["airport"] = airport
            self._json_response(200, data)

        elif path == "/api/health":
            airport = _parse_airport(qs)
            if not airport:
                self._json_response(400, {"error": "no airport specified"})
                return
            st = get_airport_state(airport)
            self._json_response(200, {
                "status": "ok",
                "airport": airport,
                "poll_count": st["poll_count"],
                "last_poll": st["last_poll"],
            })

        elif path == "/api/airports":
            self._json_response(200, {"supported": list_supported()})

        elif path == "":
            self._json_response(200, {"service": "gategetter", "version": "2.0"})

        else:
            self.send_response(404)
            self._cors_headers()
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = parse_qs(parsed.query)

        if path == "/api/track":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                flight = body.get("flight", "").strip().upper()
                airport = body.get("airport", "").strip().upper() or _parse_airport(qs)
                phone = body.get("phone", "").strip()
                if not flight:
                    self._json_response(400, {"error": "missing 'flight' field"})
                    return
                if not airport:
                    self._json_response(400, {"error": "no airport specified"})
                    return
                ensure_airport(airport)
                add_tracked(airport, flight)
                if phone:
                    add_subscriber(airport, flight, phone)
                resp = {"ok": True, "tracked": flight, "airport": airport}
                if phone:
                    resp["subscribed"] = phone
                self._json_response(200, resp)
            except Exception as e:
                self._json_response(400, {"error": str(e)})

        else:
            self.send_response(404)
            self._cors_headers()
            self.end_headers()

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = parse_qs(parsed.query)

        if path == "/api/track":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                flight = body.get("flight", "").strip().upper()
                airport = body.get("airport", "").strip().upper() or _parse_airport(qs)
                if not airport:
                    self._json_response(400, {"error": "no airport specified"})
                    return
                remove_tracked(airport, flight)
                self._json_response(200, {"ok": True, "removed": flight, "airport": airport})
            except Exception as e:
                self._json_response(400, {"error": str(e)})

        else:
            self.send_response(404)
            self._cors_headers()
            self.end_headers()

    def log_message(self, format, *args):
        pass


def _background_poller():
    """Periodically scrape airports that have tracked flights, triggering change notifications."""
    interval = int(os.environ.get("POLL_INTERVAL_SECONDS", "120"))
    print(f"[Poller] Started — polling every {interval}s when flights are tracked")
    while not shutdown_event.wait(timeout=interval):
        with state_lock:
            airports = [code for code, st in airports_state.items() if st.get("tracked")]
        if not airports:
            continue
        for airport in airports:
            try:
                print(f"[Poller] Scraping {airport} ({len(airports_state.get(airport, {}).get('tracked', []))} tracked flights)")
                scrape_once(airport)
            except Exception as e:
                print(f"[Poller] Error scraping {airport}: {e}")
    print("[Poller] Stopped")


def main():
    import subprocess
    import threading

    print("=" * 55)
    print("  GATEGETTER — ON-DEMAND FLIGHT SCRAPER (API)")
    print("=" * 55)

    if not os.environ.get("RAILWAY_ENVIRONMENT"):
        subprocess.run(f"lsof -ti:{PORT} | xargs kill -9 2>/dev/null", shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)

    poller = threading.Thread(target=_background_poller, daemon=True, name="gategetter-poller")
    poller.start()

    server = ReuseHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\n  API: http://localhost:{PORT}")
    print(f"  Endpoints:")
    print(f"    GET  /api/data?airport=OAK    — scrape and return flights")
    print(f"    GET  /api/health?airport=OAK")
    print(f"    GET  /api/airports")
    print(f"    POST /api/track               {{\"flight\": \"UA123\", \"airport\": \"OAK\"}}")
    print(f"    DELETE /api/track             {{\"flight\": \"UA123\", \"airport\": \"OAK\"}}")
    print()

    def shutdown(sig, frame):
        print("\n\nShutting down...")
        shutdown_event.set()
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    server.serve_forever()


if __name__ == "__main__":
    main()
