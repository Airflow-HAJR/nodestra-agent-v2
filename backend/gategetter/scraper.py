"""
Scraper: performs a single on-demand scrape for an airport.
Called directly by the API handler — no background loop.
"""

import time
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from airports import build_url
from scrape_test import ROW_SELECTOR, extract_flight_from_row, scroll_to_load_all, check_for_blocking
from diffing import diff_flights
from notifications import build_notifications
from state import airports_state, state_lock, ensure_airport
from supabase_cleanup import cleanup_departed_flight, upsert_flights
from auto_notify import notify_subscribers_of_changes


def scrape_once(airport_code: str) -> None:
    """
    Perform a single scrape for the given airport and update shared state.
    Launches a browser, loads the departures page, extracts flights, runs
    change detection, syncs to Supabase, and triggers subscriber notifications
    if significant changes are found.
    """
    airport_code = airport_code.upper()
    url = build_url(airport_code)
    ensure_airport(airport_code)
    now = datetime.now()
    print(f"[{airport_code}] [{now.strftime('%H:%M:%S')}] Scraping...")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            viewport={"width": 1440, "height": 900},
        )
        page = context.new_page()
        try:
            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
            except PlaywrightTimeout:
                pass
            time.sleep(3)

            blocking = check_for_blocking(page)
            if blocking:
                for b in blocking:
                    print(f"  [{airport_code}] WARNING: {b}")

            scroll_to_load_all(page)

            rows = page.query_selector_all(ROW_SELECTOR)
            flights = []
            for row in rows:
                f = extract_flight_from_row(row)
                if f:
                    flights.append(f)
        finally:
            browser.close()

    with state_lock:
        prev_flights = list(airports_state[airport_code].get("flights", []))

    upsert_flights(flights, airport_code)
    new_changes = diff_flights(prev_flights, flights) if prev_flights else []

    if new_changes:
        print(f"  [{airport_code}] {len(new_changes)} change(s):")
        for c in new_changes:
            print(f"    {c['detail']}")
        notify_subscribers_of_changes(new_changes, airport_code)

    new_notifs = build_notifications(new_changes)

    with state_lock:
        st = airports_state[airport_code]
        st["flights"] = flights
        st["notifications"] = new_notifs + st["notifications"]
        st["changes"] = new_changes + st["changes"]
        st["last_poll"] = now.strftime("%H:%M:%S")
        st["poll_count"] += 1
        st["status"] = "ok"

        curr_by_fn = {f["flight_number"]: f for f in flights if f.get("flight_number")}
        all_departed = [
            fn for fn, fdata in curr_by_fn.items()
            if fdata.get("status") and "depart" in fdata["status"].lower()
        ]
        for fn in all_departed:
            if fn in st["tracked"]:
                st["tracked"].remove(fn)
                st.get("subscribers", {}).pop(fn, None)
                print(f"  [{airport_code}] {fn} departed — unpinned")

    for fn in all_departed:
        cleanup_departed_flight(fn, airport_code)

    print(f"  [{airport_code}] {len(flights)} flights scraped.")
