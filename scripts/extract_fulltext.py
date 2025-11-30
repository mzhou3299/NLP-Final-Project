import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
import os

RAW_PATH = "data/raw/news_filtered_20251129_161820.json"
OUT_PATH = "data/articles/fulltext.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
}

# Common junk patterns we want to remove
BAD_PHRASES = [
    "Subscribe now",
    "Sign up",
    "Sign up for free",
    "Get this delivered",
    "By continuing to use our site",
    "Advertisement",
    "Continue reading",
    "Global Business and Financial News",
    "Data is a real-time snapshot",
    "Read more aboutcookies",
    "Unauthorized distribution",
    "You can manage saved articles",
    "©",
]

def clean_paragraph(text):
    if not text:
        return ""
    for bad in BAD_PHRASES:
        if bad.lower() in text.lower():
            return ""
    return text.strip()

def extract_text(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")

        # Try multiple containers (balanced approach)
        candidates = []

        # Basic <p> tags
        candidates.extend([p.get_text(" ", strip=True) for p in soup.find_all("p")])

        # Article or section blocks
        for tag in ["article", "section"]:
            for block in soup.find_all(tag):
                candidates.extend(block.get_text(" ", strip=True).split("\n"))

        # Remove garbage lines
        cleaned = [clean_paragraph(c) for c in candidates if clean_paragraph(c)]
        cleaned = [c for c in cleaned if len(c.split()) > 5]  # drop fragments

        return "\n".join(cleaned)

    except Exception as e:
        print("[ERROR] extracting:", url, e)
        return ""

def load_raw():
    with open(RAW_PATH) as f:
        return json.load(f)

def main():
    articles = load_raw()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with open(OUT_PATH, "w") as f:
        f.write("url,title,text\n")

        for art in articles:
            url = art.get("url")
            title = (art.get("title") or "").replace(",", " ")
            print("[EXTRACT]", url)

            fulltext = extract_text(url)
            fulltext_csv = fulltext.replace("\n", " ").replace(",", " ")

            f.write(f"{url},{title},{fulltext_csv}\n")
            time.sleep(0.5)

    print("[DONE] Extracted full text for all articles.")

if __name__ == "__main__":
    main()
