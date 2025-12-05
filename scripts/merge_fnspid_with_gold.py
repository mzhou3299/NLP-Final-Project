import pandas as pd
import re


FN_PATH = "data/external/fnspid_clean.csv"
SENT_PATH = "data/sentences/sentences_clean.csv"
OUT_PATH = "data/sentences/sentences_final_clean.csv"


def split_into_sentences(text):
    if not isinstance(text, str):
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if len(p.strip().split()) >= 5]


JUNK_PATTERNS = [
    r"Got a confidential news tip",
    r"Subscribe today",
    r"Taken from CNBC",
    r"Daily Open",
    r"newsletter",
    r"CNBC.?s Daily Open",
    r"CNBC.?s international markets newsletter",

    r"Analyst.?s Disclosure",
    r"Seeking Alpha",
    r"Past performance is no guarantee",
    r"Our analysts are third party authors",
    r"Any views or opinions expressed",
    r"I wrote this article myself",
    r"I/we have a beneficial",
    r"Disclosure:",
    
    r"Read here for the full list",
    r"Check out the companies making the biggest moves",
    r"-- CNBC.?s.*contributed to this report",
    
    r"^WATCH:",
    r"^WATCH\b",
    r"^LONDON —",
    r"^NEW YORK —",
    r"^HONG KONG —",

    r"^Got a confidential",
    r"^U\.S\. futures are mostly flat",
    r"^The stock market will close early",
]

def is_junk(sentence):
    if not isinstance(sentence, str):
        return True
    for pat in JUNK_PATTERNS:
        if re.search(pat, sentence, flags=re.IGNORECASE):
            return True
    return False

sent_df = pd.read_csv(SENT_PATH)

fn_df = pd.read_csv(FN_PATH)
fn_df["sentence"] = fn_df["sentence"].astype(str).str.strip()

rows = []

for _, row in fn_df.iterrows():
    url = row.get("url", None)
    raw_text = row["sentence"]

    # break into sentences
    pieces = split_into_sentences(raw_text)

    for s in pieces:
        if len(s) < 5:
            continue
        if is_junk(s):
            continue

        rows.append({"url": url, "sentence": s})

fn_sent_df = pd.DataFrame(rows)


merged = pd.concat([sent_df, fn_sent_df], ignore_index=True)

merged = merged.drop_duplicates(subset=["sentence"]).reset_index(drop=True)


merged.to_csv(OUT_PATH, index=False)

print("\nDONE: wrote →", OUT_PATH)
print("Original sentences:", len(sent_df))
print("New fnspid sentences added after cleaning:", len(fn_sent_df))
print("Final merged total:", len(merged))
