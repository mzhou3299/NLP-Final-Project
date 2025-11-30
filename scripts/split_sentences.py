import csv
import re
import os

IN_PATH = "data/articles/fulltext.csv"
OUT_PATH = "data/sentences/sentences.csv"

# Extra cleaning patterns
JUNK_PATTERNS = [
    r"Global Business and Financial News.*",
    r"Data is a real-time snapshot.*",
    r"Read more aboutcookies.*",
    r"© 20\d\d.*",
    r"Unauthorized distribution.*",
    r"Subscribe now.*",
    r"By continuing to use our site.*",
    r"Advertisement.*",
    r"Continue reading.*",
]

def clean_text(t):
    if not t:
        return ""

    for pat in JUNK_PATTERNS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def split_into_sentences(text):
    # Balanced regex that handles most cases without spacy
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if len(p.strip().split()) >= 5]

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out = []

    with open(IN_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row["url"]
            title = clean_text(row["title"])
            text = clean_text(row["text"])

            if not text:
                continue

            sentences = split_into_sentences(text)

            for s in sentences:
                out.append([url, s])

    with open(OUT_PATH, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "sentence"])
        for row in out:
            writer.writerow(row)

    print("[DONE] Split into sentences:", len(out))

if __name__ == "__main__":
    main()
