"""
prepare_training_data.py - Prepare large training dataset for FinBERT.

Splits FNSPID articles into sentences, labels using hedge keywords,
and creates train/val/test splits.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
FNSPID_PATH = "data/external/fnspid_clean.csv"
GOLD_DEV_PATH = "data/splits/dev.csv"
GOLD_TEST_PATH = "data/splits/test.csv"
OUTPUT_DIR = "data/splits"

RANDOM_STATE = 42

# Hedge keywords for labeling
HEDGE_KEYWORDS = [
    'may', 'might', 'could', 'would', 'possibly', 'perhaps', 'likely', 'unlikely',
    'appear', 'seem', 'suggest', 'indicate', 'estimate', 'forecast', 'expect',
    'expected', 'project', 'potential', 'approximately', 'around', 'about', 'roughly',
    'believed', 'appears', 'seems', 'reported', 'reportedly', 'alleged',
    'allegedly', 'rumored', 'speculation', 'speculate', 'anticipated',
    'probably', 'presumably', 'poised', 'set to', 'plans to', 'aims to',
    'seeks to', 'intends'
]

# Compile regex for word boundary matching
HEDGE_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(kw) for kw in HEDGE_KEYWORDS) + r')\b',
    re.IGNORECASE
)


def split_into_sentences(text):
    """Split text into sentences using regex."""
    # Split on sentence-ending punctuation followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return sentences


def label_sentence(sentence):
    """Label a sentence as hedge (1) or non-hedge (0) based on keywords."""
    if HEDGE_PATTERN.search(sentence):
        return 1
    return 0


def clean_sentence(sentence):
    """Clean and validate a sentence."""
    # Remove extra whitespace
    sentence = ' '.join(sentence.split())

    # Remove sentences that are too short
    if len(sentence.split()) < 5:
        return None

    # Remove sentences that are too long (likely parsing errors)
    if len(sentence.split()) > 100:
        return None

    # Remove junk patterns (disclaimers, headers, etc.)
    junk_patterns = [
        r'^Sign up',
        r'^Subscribe',
        r'^Click here',
        r'views and opinions',
        r'not necessarily reflect',
        r'Disclosure Policy',
        r'^More Stock Market News',
        r'^You can see more',
        r'^On the date of publication',
    ]

    for pattern in junk_patterns:
        if re.search(pattern, sentence, re.IGNORECASE):
            return None

    return sentence


def process_fnspid():
    """Process FNSPID dataset into labeled sentences."""
    print("[1/4] Loading FNSPID dataset...")
    df = pd.read_csv(FNSPID_PATH)
    print(f"      Loaded {len(df)} articles")

    print("[2/4] Splitting articles into sentences...")
    all_sentences = []

    for _, row in df.iterrows():
        sentences = split_into_sentences(row['sentence'])

        for sent in sentences:
            cleaned = clean_sentence(sent)
            if cleaned:
                label = label_sentence(cleaned)
                all_sentences.append({
                    'sentence': cleaned,
                    'label': label
                })

    sentences_df = pd.DataFrame(all_sentences)
    print(f"      Generated {len(sentences_df)} sentences")

    # Label distribution
    hedge_count = sentences_df['label'].sum()
    non_hedge_count = len(sentences_df) - hedge_count
    print(f"      Label distribution: {hedge_count} hedge ({hedge_count/len(sentences_df)*100:.1f}%), "
          f"{non_hedge_count} non-hedge ({non_hedge_count/len(sentences_df)*100:.1f}%)")

    return sentences_df


def create_splits(sentences_df):
    """Create train/val/test splits."""
    print("[3/4] Creating stratified splits...")

    # First split: 80% train, 20% temp
    train_df, temp_df = train_test_split(
        sentences_df,
        test_size=0.2,
        stratify=sentences_df['label'],
        random_state=RANDOM_STATE
    )

    # Second split: 50% val, 50% test (from temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label'],
        random_state=RANDOM_STATE
    )

    print(f"      Train: {len(train_df)} samples")
    print(f"      Val:   {len(val_df)} samples")
    print(f"      Test:  {len(test_df)} samples")

    return train_df, val_df, test_df


def add_gold_samples(val_df, test_df):
    """Add gold-labeled samples to validation and test sets."""
    print("[4/4] Adding gold-labeled samples...")

    # Load gold data
    gold_dev = pd.read_csv(GOLD_DEV_PATH)
    gold_test = pd.read_csv(GOLD_TEST_PATH)

    # Standardize column names
    gold_dev = gold_dev[['sentence', 'gold_label']].rename(columns={'gold_label': 'label'})
    gold_test = gold_test[['sentence', 'gold_label']].rename(columns={'gold_label': 'label'})

    # Add source column to track origin
    val_df = val_df.copy()
    test_df = test_df.copy()
    gold_dev = gold_dev.copy()
    gold_test = gold_test.copy()

    val_df['source'] = 'fnspid'
    test_df['source'] = 'fnspid'
    gold_dev['source'] = 'gold'
    gold_test['source'] = 'gold'

    # Combine
    val_combined = pd.concat([val_df, gold_dev], ignore_index=True)
    test_combined = pd.concat([test_df, gold_test], ignore_index=True)

    print(f"      Val with gold:  {len(val_combined)} samples ({len(gold_dev)} gold)")
    print(f"      Test with gold: {len(test_combined)} samples ({len(gold_test)} gold)")

    return val_combined, test_combined


def main():
    print("=" * 60)
    print("PREPARE TRAINING DATA FOR FINBERT")
    print("=" * 60)

    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    # Process FNSPID
    sentences_df = process_fnspid()

    # Create splits
    train_df, val_df, test_df = create_splits(sentences_df)

    # Add gold samples
    val_df, test_df = add_gold_samples(val_df, test_df)

    # Save to CSV
    train_path = f"{OUTPUT_DIR}/train_large.csv"
    val_path = f"{OUTPUT_DIR}/val_large.csv"
    test_path = f"{OUTPUT_DIR}/test_large.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Train set: {len(train_df):,} samples -> {train_path}")
    print(f"Val set:   {len(val_df):,} samples -> {val_path}")
    print(f"Test set:  {len(test_df):,} samples -> {test_path}")

    # Final label distributions
    print("\nLabel distributions:")
    print(f"  Train: {train_df['label'].mean()*100:.1f}% hedge")
    print(f"  Val:   {val_df['label'].mean()*100:.1f}% hedge")
    print(f"  Test:  {test_df['label'].mean()*100:.1f}% hedge")

    print("\n[DONE] Data preparation complete!")


if __name__ == "__main__":
    main()
