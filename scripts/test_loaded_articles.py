import json
from urllib.parse import urlparse

RAW_FILE = "data/raw/news_raw.json"

ALLOWED_DOMAINS = {
    "marketwatch.com",
    "reuters.com",
    "cnbc.com",
    "finance.yahoo.com",
    "bloomberg.com",
    "businessinsider.com",
    "financialpost.com",
}

TECH_KEYWORDS = [
    "apple", "microsoft", "amazon", "google", "alphabet",
    "nvidia", "tesla", "chip", "semiconductor", "ai",
    "earnings", "cloud", "meta", "intel", "salesforce",
]

def looks_tech_related(title, description):
    # Handle None values safely
    title = title or ""
    description = description or ""
    text = (title + " " + description).lower()
    return any(k in text for k in TECH_KEYWORDS)

def main():
    print("[TEST] Loading articles...")
    try:
        with open(RAW_FILE, "r") as f:
            articles = json.load(f)
    except FileNotFoundError:
        print("[ERROR] Could not find:", RAW_FILE)
        return

    print(f"[INFO] Loaded {len(articles)} articles\n")

    bad_domain = 0
    not_tech = 0
    empty_fields = 0
    valid = 0

    for art in articles:
        url = art.get("url")
        title = art.get("title")
        desc = art.get("description")

        # Check valid fields
        if not url or not title:
            empty_fields += 1
            continue

        # Check domain
        domain = urlparse(url).netloc.replace("www.", "")
        if domain not in ALLOWED_DOMAINS:
            bad_domain += 1

        # Check tech relevance
        if looks_tech_related(title, desc):
            valid += 1
        else:
            not_tech += 1

    print("===== TEST RESULTS =====")
    print(f"Total articles: {len(articles)}")
    print(f"✔ Tech-related = {valid}")
    print(f"✘ Non-tech = {not_tech}")
    print(f"✘ Wrong domain = {bad_domain}")
    print(f"✘ Missing title/url = {empty_fields}")
    print("========================")

    if bad_domain == 0:
        print("✓ Domain filtering looks correct.")
    else:
        print("⚠ Some articles came from domains not in your allowlist.")

    if empty_fields == 0:
        print("✓ All articles have title & URL.")
    else:
        print("⚠ Some articles are malformed but were skipped.")

    if valid > 0:
        print("✓ Scraper successfully fetched tech-market news.")
    else:
        print("❌ Something is wrong — no tech articles detected!")

if __name__ == "__main__":
    main()
