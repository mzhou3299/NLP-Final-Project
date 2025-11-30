import csv
import re
import os

IN_PATH = "data/sentences/sentences.csv"
OUT_PATH = "data/annotations/to_annotate.csv"

# Medium-size hedge cue list
HEDGE_CUES = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bpossibly\b",
    r"\bpotentially\b",
    r"\blikely\b",
    r"\bunlikely\b",
    r"\bappears\b",
    r"\bseems\b",
    r"\bsuggests\b",
    r"\bindicates\b",
    r"\bestimates?\b",
    r"\bexpected\b",
    r"\bprojected\b",
    r"\buncertain\b",
    r"\brisk\b",
    r"\bdepends on\b",
]

pattern = re.compile("|".join(HEDGE_CUES), flags=re.IGNORECASE)

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    rows = []
    with open(IN_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sent = row["sentence"]
            if pattern.search(sent):
                rows.append([row["url"], sent])

    # Limit to ~200 for manual annotation
    rows = rows[:200]

    with open(OUT_PATH, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "sentence"])
        for r in rows:
            writer.writerow(r)

    print("[DONE] Filtered hedged sentences:", len(rows))

if __name__ == "__main__":
    main()
