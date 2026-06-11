"""
Flighty.com Departures Scrape Test

Opens the departures page in a headless browser, scrolls to load all flights,
extracts flight data from the DOM, and reports on scrapability.

Usage:
    python3 scrape_test.py          # defaults to SJC
    python3 scrape_test.py SFO      # test a specific airport
"""

import json
import csv
import sys
import time
import os
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from airports import build_url
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MAX_SCROLL_ATTEMPTS = 100
SCROLL_PAUSE = 1.5  # seconds between scrolls

# Each flight row is a div with h-[48px] and grid-cols-[...] classes
ROW_SELECTOR = 'div[class*="h-[48px]"][class*="grid-cols-"]'


def extract_flight_from_row(row):
    """Extract flight data from a single row element.

    Scrolls the row into view first to force content-visibility: auto to render.
    """
    row.scroll_into_view_if_needed()

    children = row.query_selector_all(':scope > *')
    texts = [c.inner_text().strip() for c in children]

    if len(texts) < 6:
        return None

    # Parse destination: "City\n\nIATA" -> separate fields
    dest_raw = texts[3] if len(texts) > 3 else ""
    dest_parts = [p.strip() for p in dest_raw.split("\n") if p.strip()]
    destination_city = dest_parts[0] if dest_parts else None
    destination_iata = dest_parts[1] if len(dest_parts) > 1 else None

    flight = {
        "scheduled_time": texts[0] if len(texts) > 0 else None,
        "actual_time": texts[1] if len(texts) > 1 else None,
        "flight_number": texts[2] if len(texts) > 2 else None,
        "destination_city": destination_city,
        "destination_iata": destination_iata,
        "airline": texts[4] if len(texts) > 4 else None,
        "terminal": texts[5] if len(texts) > 5 else None,
        "gate": texts[6] if len(texts) > 6 else None,
        "status": texts[7] if len(texts) > 7 else None,
    }

    # Clean up empty strings to None
    return {k: (v if v else None) for k, v in flight.items()}


def scroll_to_load_all(page):
    """Scroll down repeatedly to trigger lazy-loading of more flights."""
    prev_count = 0
    stable_rounds = 0

    for i in range(MAX_SCROLL_ATTEMPTS):
        rows = page.query_selector_all(ROW_SELECTOR)
        current_count = len(rows)

        if i % 5 == 0 or current_count != prev_count:
            print(f"  Scroll {i + 1}: {current_count} flights loaded")

        if current_count == prev_count:
            stable_rounds += 1
            if stable_rounds >= 5:
                print(f"  No new flights after {stable_rounds} scrolls. Done.")
                break
        else:
            stable_rounds = 0

        prev_count = current_count

        # Scroll the last flight row into view to trigger IntersectionObserver
        if rows:
            rows[-1].scroll_into_view_if_needed()
        # Also scroll to absolute bottom
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(SCROLL_PAUSE)

    return prev_count


def check_for_blocking(page):
    """Check if the page shows signs of blocking."""
    issues = []

    title = page.title().lower()
    if any(word in title for word in ["captcha", "challenge", "blocked", "denied", "cloudflare"]):
        issues.append(f"Suspicious page title: {page.title()}")

    body_text = page.inner_text("body").lower()
    blocking_keywords = ["captcha", "are you a robot", "access denied", "rate limit",
                         "too many requests", "please verify", "blocked"]
    for keyword in blocking_keywords:
        if keyword in body_text:
            issues.append(f"Blocking indicator found: '{keyword}'")

    current_url = page.url
    if "challenge" in current_url or "captcha" in current_url:
        issues.append(f"Redirected to challenge page: {current_url}")

    return issues


def save_results(flights, airport_code="SJC"):
    """Save flight data to JSON and CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    code_lower = airport_code.lower()

    json_path = os.path.join(OUTPUT_DIR, f"{code_lower}_departures.json")
    with open(json_path, "w") as f:
        json.dump(flights, f, indent=2)
    print(f"  Saved {len(flights)} flights to {json_path}")

    csv_path = os.path.join(OUTPUT_DIR, f"{code_lower}_departures.csv")
    fieldnames = ["scheduled_time", "actual_time", "flight_number",
                   "destination_city", "destination_iata",
                   "airline", "terminal", "gate", "status"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flights)
    print(f"  Saved {len(flights)} flights to {csv_path}")


def print_report(flights, blocking_issues, total_loaded):
    """Print a summary report."""
    print("\n" + "=" * 60)
    print("  FLIGHTY.COM SCRAPING TEST REPORT")
    print("=" * 60)

    print(f"\nFlights loaded in browser: {total_loaded}")
    print(f"Flights extracted:         {len(flights)}")

    if flights:
        fields = ["scheduled_time", "actual_time", "flight_number",
                   "destination_city", "destination_iata",
                   "airline", "terminal", "gate", "status"]
        print("\nData completeness:")
        for field in fields:
            present = sum(1 for f in flights if f.get(field))
            pct = (present / len(flights) * 100) if flights else 0
            print(f"  {field:20s}: {present:4d}/{len(flights)} ({pct:.0f}%)")

    print(f"\nBlocking detection:")
    if blocking_issues:
        for issue in blocking_issues:
            print(f"  WARNING: {issue}")
    else:
        print("  No blocking detected")

    print(f"\nVerdict: ", end="")
    if blocking_issues:
        print("BLOCKED or PARTIALLY BLOCKED")
    elif len(flights) == 0:
        print("FAILED - No data extracted (may need selector updates)")
    elif len(flights) < 25:
        print("PARTIALLY SCRAPABLE - Fewer flights than expected")
    else:
        print("SCRAPABLE")

    print("=" * 60)


def dump_page_debug(page):
    """Dump page HTML and text for debugging."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    debug_path = os.path.join(OUTPUT_DIR, "page_debug.html")
    with open(debug_path, "w") as f:
        f.write(page.content())

    debug_txt_path = os.path.join(OUTPUT_DIR, "page_text.txt")
    with open(debug_txt_path, "w") as f:
        f.write(page.inner_text("body"))


def main():
    airport_code = sys.argv[1].upper() if len(sys.argv) > 1 else "SJC"
    url = build_url(airport_code)

    print(f"Flighty.com {airport_code} Departures Scrape Test")
    print(f"Target: {url}\n")

    with sync_playwright() as p:
        print("Launching browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            viewport={"width": 1440, "height": 900},
        )
        page = context.new_page()

        print("Loading page...")
        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
        except PlaywrightTimeout:
            print("  Page load timed out (30s) - continuing with what loaded")

        print("Checking for blocking...")
        blocking_issues = check_for_blocking(page)
        if blocking_issues:
            for issue in blocking_issues:
                print(f"  WARNING: {issue}")

        # Wait for JS rendering
        time.sleep(3)

        initial_rows = page.query_selector_all(ROW_SELECTOR)
        print(f"Initial flights rendered: {len(initial_rows)}")

        # Scroll to load all flights
        print("\nScrolling to load all flights...")
        total_loaded = scroll_to_load_all(page)

        # Extract flights — scroll each row into view to handle content-visibility
        print("\nExtracting flight data (scrolling each row into view)...")
        rows = page.query_selector_all(ROW_SELECTOR)
        flights = []
        for i, row in enumerate(rows):
            flight = extract_flight_from_row(row)
            if flight:
                flights.append(flight)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(rows)} rows...")

        print(f"  Extracted {len(flights)} flights from {len(rows)} rows")

        if flights:
            print(f"\n  Sample flights:")
            for f in flights[:3]:
                fn = f['flight_number'] or '?'
                dc = f['destination_city'] or '?'
                di = f['destination_iata'] or '?'
                st = f['scheduled_time'] or '?'
                tm = f['terminal'] or '-'
                gt = f['gate'] or '-'
                ss = f['status'] or '?'
                print(f"    {fn:10s} -> {dc:15s} ({di})  @ {st}  T{tm}  Gate {gt}  [{ss}]")

        dump_page_debug(page)

        print("\nSaving results...")
        save_results(flights, airport_code)

        print_report(flights, blocking_issues, total_loaded)

        browser.close()


if __name__ == "__main__":
    main()
