"""
split_data.py - Create stratified train/dev/test splits for hedge detection.

Strategy: 80/20 stratified split (dev/test) with 5-fold CV indices for dev set.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
import json

DATA_PATH = "data/annotations/completed_annotation_clean.csv"
SPLITS_DIR = "data/splits"
RANDOM_STATE = 42


def main():
    # Load annotated data
    df = pd.read_csv(DATA_PATH)
    print(f"[LOAD] Loaded {len(df)} annotated sentences")

    # Show class distribution
    label_counts = df['gold_label'].value_counts()
    print(f"[INFO] Class distribution:")
    print(f"       Non-hedge (0): {label_counts.get(0, 0)}")
    print(f"       Hedge (1): {label_counts.get(1, 0)}")

    # Stratified split: 80% dev, 20% test
    dev_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['gold_label'],
        random_state=RANDOM_STATE
    )

    print(f"\n[SPLIT] Dev set: {len(dev_df)} samples")
    print(f"[SPLIT] Test set: {len(test_df)} samples")

    # Generate 5-fold CV indices for dev set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_indices = []

    dev_labels = dev_df['gold_label'].values
    dev_indices = dev_df.index.tolist()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dev_indices, dev_labels)):
        fold_indices.append({
            'fold': fold_idx,
            'train_indices': [dev_indices[i] for i in train_idx],
            'val_indices': [dev_indices[i] for i in val_idx]
        })
        print(f"[FOLD {fold_idx}] Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Save splits
    Path(SPLITS_DIR).mkdir(exist_ok=True, parents=True)

    dev_df.to_csv(f"{SPLITS_DIR}/dev.csv", index=False)
    test_df.to_csv(f"{SPLITS_DIR}/test.csv", index=False)

    with open(f"{SPLITS_DIR}/fold_indices.json", 'w') as f:
        json.dump(fold_indices, f, indent=2)

    print(f"\n[DONE] Saved splits to {SPLITS_DIR}/")
    print(f"       - dev.csv ({len(dev_df)} samples)")
    print(f"       - test.csv ({len(test_df)} samples)")
    print(f"       - fold_indices.json (5 folds)")


if __name__ == "__main__":
    main()
