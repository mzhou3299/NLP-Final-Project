import csv
import random
import pandas as pd
from pathlib import Path

RAW_SENT_PATH = "data/sentences/sentences.csv"   # <-- FIXED HERE
OUT_PATH = "data/annotations/to_annotate.csv"

HEDGE_CUES = [
    "may", "might", "could", "likely", "unlikely", "possibly",
    "expects", "expect", "forecast", "project", "anticipate",
    "suggests", "appears", "seems", "believe", "estimates",
    "probable", "potential", "could see", "may see", "risk",
    "uncertain", "indicate", "signals"
]

def looks_hedgy(s):
    s_low = s.lower()
    return any(h in s_low for h in HEDGE_CUES)

def main():
    df = pd.read_csv(RAW_SENT_PATH)
    print(f"[LOAD] Loaded {len(df)} total sentences")

    df = df.dropna(subset=["sentence"])
    df["sentence"] = df["sentence"].str.strip()
    df = df[df["sentence"].str.len() > 10]  # remove tiny fragments

    hedgy = df[df["sentence"].apply(looks_hedgy)]
    non_hedgy = df[~df["sentence"].apply(looks_hedgy)]

    print(f"[INFO] Hedgy candidates: {len(hedgy)}")
    print(f"[INFO] Non-hedgy candidates: {len(non_hedgy)}")

    hedgy_sample = hedgy.sample(n=min(50, len(hedgy)), random_state=42)
    non_sample = non_hedgy.sample(n=min(50, len(non_hedgy)), random_state=42)

    sampled = pd.concat([hedgy_sample, non_sample]).sample(frac=1, random_state=42)

    rows = []
    for i, row in sampled.iterrows():
        rows.append({
            "id": f"S{i}",
            "sentence": row["sentence"],
            "url": row["url"],
            "rule_pred": 1 if looks_hedgy(row["sentence"]) else 0,
            "gold_label": "?"
        })

    out_df = pd.DataFrame(rows)
    Path("data/annotations").mkdir(exist_ok=True, parents=True)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"[DONE] Wrote {len(out_df)} sentences → {OUT_PATH}")

if __name__ == "__main__":
    main()
