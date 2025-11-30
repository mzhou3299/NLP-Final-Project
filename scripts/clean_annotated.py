import pandas as pd
import re

# ========= PATHS ===========
ANNOT_PATH = "data/annotations/completed_annotation.csv"
SENT_PATH = "data/sentences/sentences.csv"

OUT_ANNOT = "data/annotations/completed_annotation_clean.csv"
OUT_SENT = "data/sentences/sentences_clean.csv"

# ========= LOAD DATA ===========
print("[LOAD] Loading annotation & sentence datasets...")
annot = pd.read_csv(ANNOT_PATH)
sentences = pd.read_csv(SENT_PATH)

# ========= CLEAN ANNOTATION CSV ===========
print("[CLEAN] Cleaning gold annotations...")

# Remove rule_pred column if present
if "rule_pred" in annot.columns:
    annot = annot.drop(columns=["rule_pred"])

# Remove rows labeled "delete"
annot = annot[~annot["gold_label"].str.lower().eq("delete")]

# Convert gold_label to integer
annot["gold_label"] = annot["gold_label"].astype(int)

# Drop duplicates based on sentence text
annot = annot.drop_duplicates(subset=["sentence"])

print(f"[CLEAN] Gold annotations now contain {len(annot)} rows.")

# ========= LOAD DELETE SENTENCES ===========
# These are all sentences removed from the gold set.
deleted_sentences = pd.read_csv(ANNOT_PATH)
deleted_sentences = deleted_sentences[deleted_sentences["gold_label"] == "delete"]["sentence"].tolist()

# ========= CLEAN SENTENCE DATASET ===========
print("[CLEAN] Cleaning full sentence dataset...")

sent_clean = sentences.copy()

# --- Remove duplicates in the raw dataset ---
sent_clean = sent_clean.drop_duplicates(subset=["sentence"])

# --- Remove deleted sentences (exact matches) ---
sent_clean = sent_clean[~sent_clean["sentence"].isin(deleted_sentences)]

# ========= REMOVE BOILERPLATE / JUNK SENTENCES ===========
JUNK_PATTERNS = [

    # CNBC / newsletter boilerplate
    r"Got a confidential news tip",
    r"Subscribe today",
    r"Taken from CNBC",
    r"Daily Open",
    r"newsletter",
    r"CNBC.?s Daily Open",
    r"CNBC.?s international markets newsletter",

    # SeekingAlpha disclosures
    r"Analyst.?s Disclosure",
    r"Seeking Alpha",
    r"Past performance is no guarantee",
    r"Our analysts are third party authors",
    r"Any views or opinions expressed",
    r"I wrote this article myself",
    r"I/we have a beneficial",
    r"Disclosure:",

    # Article headers / footers / non-content meta
    r"Read here for the full list",
    r"Check out the companies making the biggest moves",
    r"-- CNBC.?s.*contributed to this report",

    # Pure stock-ticker boilerplate
    r"^WATCH:",
    r"^WATCH\b",
    r"WATCH:",

    # Ultra-short contextual intros
    r"^LONDON —",
    r"^NEW YORK —",
    r"^HONG KONG —",

    # Very generic irrelevant lines
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

sent_clean = sent_clean[~sent_clean["sentence"].apply(is_junk)]

# ========= RESET INDEX ===========
sent_clean = sent_clean.reset_index(drop=True)
annot = annot.reset_index(drop=True)

# ========= ALIGN SENTENCE DATASET WITH ANNOTATIONS ===========

# Find annotation sentences missing from the cleaned sentence set
missing = annot[~annot["sentence"].isin(sent_clean["sentence"])]

# Add them back
if len(missing) > 0:
    print(f"[FIX] Adding {len(missing)} missing sentences back into sent_clean...")
    sent_clean = pd.concat([sent_clean, missing[["url", "sentence"]]], ignore_index=True)

# Drop duplicates just in case
sent_clean = sent_clean.drop_duplicates(subset=["sentence"])

# ========= EXPORT ===========
annot.to_csv(OUT_ANNOT, index=False)
sent_clean.to_csv(OUT_SENT, index=False)

print("[DONE] Cleaning complete!")
print(f"Saved gold annotations → {OUT_ANNOT}")
print(f"Saved cleaned sentences → {OUT_SENT}")
