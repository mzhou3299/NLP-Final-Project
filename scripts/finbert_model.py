"""
finbert_model.py - FinBERT fine-tuning for hedge detection.

Uses the larger training dataset prepared by prepare_training_data.py.
FinBERT is pre-trained on financial text, making it well-suited for this task.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Paths
TRAIN_PATH = "data/splits/train_large.csv"
VAL_PATH = "data/splits/val_large.csv"
TEST_PATH = "data/splits/test_large.csv"
MODELS_DIR = "models/finbert"
RESULTS_DIR = "results"

# Model configuration
MODEL_NAME = "ProsusAI/finbert"  # FinBERT - pre-trained on financial text
MAX_LENGTH = 128
BATCH_SIZE = 16  # Larger batch for bigger dataset
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5  # Fewer epochs needed with more data
EARLY_STOPPING_PATIENCE = 2
RANDOM_STATE = 42


class HedgeDataset(Dataset):
    """Custom dataset for hedge detection."""

    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Compute metrics for Trainer."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0)
    }


def evaluate_predictions(y_true, y_pred):
    """Calculate detailed metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def evaluate_on_gold(trainer, test_df, tokenizer):
    """Evaluate specifically on gold-labeled samples."""
    gold_df = test_df[test_df['source'] == 'gold']

    if len(gold_df) == 0:
        return None

    gold_sentences = gold_df['sentence'].tolist()
    gold_labels = gold_df['label'].values

    gold_dataset = HedgeDataset(gold_sentences, gold_labels.tolist(), tokenizer, MAX_LENGTH)
    predictions = trainer.predict(gold_dataset)
    preds = np.argmax(predictions.predictions, axis=1)

    return evaluate_predictions(gold_labels, preds)


def main():
    # Check for GPU/MPS
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load data
    print("\n[LOADING] Training data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df['sentence'].tolist()
    y_train = train_df['label'].values
    X_val = val_df['sentence'].tolist()
    y_val = val_df['label'].values
    X_test = test_df['sentence'].tolist()
    y_test = test_df['label'].values

    print("=" * 60)
    print("FINBERT MODEL: Fine-Tuning for Hedge Detection")
    print("=" * 60)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Train samples: {len(X_train):,}")
    print(f"Val samples:   {len(X_val):,}")
    print(f"Test samples:  {len(X_test):,}")

    # Calculate class weights for imbalanced data
    n_samples = len(y_train)
    n_hedge = y_train.sum()
    n_non_hedge = n_samples - n_hedge
    weight_non_hedge = n_samples / (2 * n_non_hedge)
    weight_hedge = n_samples / (2 * n_hedge)
    print(f"\nClass weights: non-hedge={weight_non_hedge:.3f}, hedge={weight_hedge:.3f}")

    # Load tokenizer
    print("\n[LOADING] FinBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create datasets
    print("[CREATING] Datasets...")
    train_dataset = HedgeDataset(X_train, y_train.tolist(), tokenizer, MAX_LENGTH)
    val_dataset = HedgeDataset(X_val, y_val.tolist(), tokenizer, MAX_LENGTH)
    test_dataset = HedgeDataset(X_test, y_test.tolist(), tokenizer, MAX_LENGTH)

    # Create directories
    Path(MODELS_DIR).mkdir(exist_ok=True, parents=True)
    Path(RESULTS_DIR).mkdir(exist_ok=True, parents=True)

    # Load model
    print("[LOADING] FinBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=MODELS_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        logging_dir=f"{MODELS_DIR}/logs",
        logging_steps=100,
        seed=RANDOM_STATE,
        report_to="none",
        save_total_limit=2
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    )

    # Train
    print("\n[TRAINING] Starting FinBERT fine-tuning...")
    print("-" * 60)
    trainer.train()

    # Validation results
    print("\n[VALIDATION SET]")
    val_predictions = trainer.predict(val_dataset)
    val_preds = np.argmax(val_predictions.predictions, axis=1)
    val_metrics = evaluate_predictions(y_val, val_preds)

    print(f"  Accuracy:  {val_metrics['accuracy']:.3f}")
    print(f"  Precision: {val_metrics['precision']:.3f}")
    print(f"  Recall:    {val_metrics['recall']:.3f}")
    print(f"  F1 Score:  {val_metrics['f1']:.3f}")

    # Test results
    print(f"\n[TEST SET] ({len(y_test):,} samples)")
    test_predictions = trainer.predict(test_dataset)
    test_preds = np.argmax(test_predictions.predictions, axis=1)
    test_metrics = evaluate_predictions(y_test, test_preds)

    print(f"  Accuracy:  {test_metrics['accuracy']:.3f}")
    print(f"  Precision: {test_metrics['precision']:.3f}")
    print(f"  Recall:    {test_metrics['recall']:.3f}")
    print(f"  F1 Score:  {test_metrics['f1']:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"            Pred 0  Pred 1")
    print(f"  Actual 0    {test_metrics['tn']:5d}   {test_metrics['fp']:5d}")
    print(f"  Actual 1    {test_metrics['fn']:5d}   {test_metrics['tp']:5d}")

    # Evaluate on gold samples only
    gold_metrics = evaluate_on_gold(trainer, test_df, tokenizer)
    if gold_metrics:
        print(f"\n[GOLD TEST SAMPLES ONLY]")
        print(f"  Accuracy:  {gold_metrics['accuracy']:.3f}")
        print(f"  Precision: {gold_metrics['precision']:.3f}")
        print(f"  Recall:    {gold_metrics['recall']:.3f}")
        print(f"  F1 Score:  {gold_metrics['f1']:.3f}")

    # Save model
    trainer.save_model(f"{MODELS_DIR}/final")
    tokenizer.save_pretrained(f"{MODELS_DIR}/final")

    # Save predictions
    test_df_with_preds = test_df.copy()
    test_df_with_preds['finbert_pred'] = test_preds
    test_df_with_preds.to_csv(f"{RESULTS_DIR}/finbert_predictions.csv", index=False)

    # Save metrics
    results = {
        'model': MODEL_NAME,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'val': val_metrics,
        'test': test_metrics,
        'gold_test': gold_metrics
    }

    with open(f"{RESULTS_DIR}/finbert_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[DONE] Model saved to {MODELS_DIR}/final")
    print(f"       Predictions saved to {RESULTS_DIR}/finbert_predictions.csv")
    print(f"       Metrics saved to {RESULTS_DIR}/finbert_metrics.json")


if __name__ == "__main__":
    main()
