"""
baseline_model.py - Rule-based keyword matching for hedge detection.

Uses the HEDGE_CUES list from build_annotation_set.py to predict hedges.
If any hedge cue is found in the sentence, predict 1 (hedge), else 0.
"""

import pandas as pd
import json
from pathlib import Path

# Hedge cues from build_annotation_set.py
HEDGE_CUES = [
    "may", "might", "could", "likely", "unlikely", "possibly",
    "expects", "expect", "forecast", "project", "anticipate",
    "suggests", "appears", "seems", "believe", "estimates",
    "probable", "potential", "could see", "may see", "risk",
    "uncertain", "indicate", "signals"
]

DEV_PATH = "data/splits/dev.csv"
TEST_PATH = "data/splits/test.csv"
RESULTS_DIR = "results"


def predict_hedge(sentence):
    """Return 1 if any hedge cue is found in the sentence, else 0."""
    sentence_lower = sentence.lower()
    for cue in HEDGE_CUES:
        if cue in sentence_lower:
            return 1
    return 0


def evaluate(y_true, y_pred):
    """Calculate accuracy, precision, recall, F1."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def main():
    # Load dev and test sets
    dev_df = pd.read_csv(DEV_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("=" * 50)
    print("BASELINE MODEL: Keyword Matching")
    print("=" * 50)
    print(f"\nHedge cues ({len(HEDGE_CUES)}): {HEDGE_CUES[:5]}...")

    # Predict on dev set
    dev_preds = [predict_hedge(s) for s in dev_df['sentence']]
    dev_labels = dev_df['gold_label'].tolist()
    dev_metrics = evaluate(dev_labels, dev_preds)

    print(f"\n[DEV SET] ({len(dev_df)} samples)")
    print(f"  Accuracy:  {dev_metrics['accuracy']:.3f}")
    print(f"  Precision: {dev_metrics['precision']:.3f}")
    print(f"  Recall:    {dev_metrics['recall']:.3f}")
    print(f"  F1 Score:  {dev_metrics['f1']:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"            Pred 0  Pred 1")
    print(f"  Actual 0    {dev_metrics['tn']:3d}     {dev_metrics['fp']:3d}")
    print(f"  Actual 1    {dev_metrics['fn']:3d}     {dev_metrics['tp']:3d}")

    # Predict on test set
    test_preds = [predict_hedge(s) for s in test_df['sentence']]
    test_labels = test_df['gold_label'].tolist()
    test_metrics = evaluate(test_labels, test_preds)

    print(f"\n[TEST SET] ({len(test_df)} samples)")
    print(f"  Accuracy:  {test_metrics['accuracy']:.3f}")
    print(f"  Precision: {test_metrics['precision']:.3f}")
    print(f"  Recall:    {test_metrics['recall']:.3f}")
    print(f"  F1 Score:  {test_metrics['f1']:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"            Pred 0  Pred 1")
    print(f"  Actual 0    {test_metrics['tn']:3d}     {test_metrics['fp']:3d}")
    print(f"  Actual 1    {test_metrics['fn']:3d}     {test_metrics['tp']:3d}")

    # Save predictions
    Path(RESULTS_DIR).mkdir(exist_ok=True, parents=True)

    test_df_with_preds = test_df.copy()
    test_df_with_preds['baseline_pred'] = test_preds
    test_df_with_preds.to_csv(f"{RESULTS_DIR}/baseline_predictions.csv", index=False)

    # Save metrics
    results = {
        'model': 'baseline_keyword',
        'dev': dev_metrics,
        'test': test_metrics
    }
    with open(f"{RESULTS_DIR}/baseline_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Saved predictions to {RESULTS_DIR}/baseline_predictions.csv")
    print(f"       Saved metrics to {RESULTS_DIR}/baseline_metrics.json")


if __name__ == "__main__":
    main()
