"""
transformer_model.py - DistilBERT fine-tuning for hedge detection.

Uses 5-fold stratified cross-validation with early stopping.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DEV_PATH = "data/splits/dev.csv"
TEST_PATH = "data/splits/test.csv"
MODELS_DIR = "models/distilbert"
RESULTS_DIR = "results"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
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


def train_fold(train_sentences, train_labels, val_sentences, val_labels, tokenizer, fold_idx):
    """Train on a single fold."""

    # Create datasets
    train_dataset = HedgeDataset(train_sentences, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = HedgeDataset(val_sentences, val_labels, tokenizer, MAX_LENGTH)

    # Load fresh model for this fold
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    # Training arguments
    output_dir = f"{MODELS_DIR}/fold_{fold_idx}"
    training_args = TrainingArguments(
        output_dir=output_dir,
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
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        seed=RANDOM_STATE,
        report_to="none",
        save_total_limit=1
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
    trainer.train()

    # Get validation predictions
    val_predictions = trainer.predict(val_dataset)
    val_preds = np.argmax(val_predictions.predictions, axis=1)

    return model, val_preds, val_predictions.metrics


def main():
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    dev_df = pd.read_csv(DEV_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_dev = dev_df['sentence'].tolist()
    y_dev = dev_df['gold_label'].values
    X_test = test_df['sentence'].tolist()
    y_test = test_df['gold_label'].values

    print("=" * 50)
    print("TRANSFORMER MODEL: DistilBERT Fine-Tuning")
    print("=" * 50)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Dev samples: {len(X_dev)}, Test samples: {len(X_test)}")

    # Load tokenizer
    print("\n[LOADING] Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create directories
    Path(MODELS_DIR).mkdir(exist_ok=True, parents=True)
    Path(RESULTS_DIR).mkdir(exist_ok=True, parents=True)

    # 5-fold cross-validation
    print("\n[TRAINING] 5-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    all_val_preds = np.zeros(len(X_dev))
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev)):
        print(f"\n--- Fold {fold_idx + 1}/5 ---")

        train_sentences = [X_dev[i] for i in train_idx]
        train_labels = [y_dev[i] for i in train_idx]
        val_sentences = [X_dev[i] for i in val_idx]
        val_labels = [y_dev[i] for i in val_idx]

        model, val_preds, metrics = train_fold(
            train_sentences, train_labels,
            val_sentences, val_labels,
            tokenizer, fold_idx
        )

        all_val_preds[val_idx] = val_preds
        fold_metrics.append(metrics)

        f1_key = 'eval_f1' if 'eval_f1' in metrics else 'test_f1'
        print(f"  Val F1: {metrics.get(f1_key, metrics.get('eval_f1', 0)):.3f}")

    # Dev set CV results
    dev_metrics = evaluate_predictions(y_dev, all_val_preds.astype(int))

    print(f"\n[DEV SET - CV] ({len(y_dev)} samples)")
    print(f"  Accuracy:  {dev_metrics['accuracy']:.3f}")
    print(f"  Precision: {dev_metrics['precision']:.3f}")
    print(f"  Recall:    {dev_metrics['recall']:.3f}")
    print(f"  F1 Score:  {dev_metrics['f1']:.3f}")

    # Train final model on full dev set and evaluate on test
    print("\n[TRAINING] Final model on full dev set...")

    train_dataset = HedgeDataset(X_dev, y_dev.tolist(), tokenizer, MAX_LENGTH)
    test_dataset = HedgeDataset(X_test, y_test.tolist(), tokenizer, MAX_LENGTH)

    final_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=f"{MODELS_DIR}/final",
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
        seed=RANDOM_STATE,
        report_to="none",
        save_total_limit=1
    )

    trainer = Trainer(
        model=final_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Test set predictions
    test_predictions = trainer.predict(test_dataset)
    test_preds = np.argmax(test_predictions.predictions, axis=1)
    test_metrics = evaluate_predictions(y_test, test_preds)

    print(f"\n[TEST SET] ({len(y_test)} samples)")
    print(f"  Accuracy:  {test_metrics['accuracy']:.3f}")
    print(f"  Precision: {test_metrics['precision']:.3f}")
    print(f"  Recall:    {test_metrics['recall']:.3f}")
    print(f"  F1 Score:  {test_metrics['f1']:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"            Pred 0  Pred 1")
    print(f"  Actual 0    {test_metrics['tn']:3d}     {test_metrics['fp']:3d}")
    print(f"  Actual 1    {test_metrics['fn']:3d}     {test_metrics['tp']:3d}")

    # Save final model
    trainer.save_model(f"{MODELS_DIR}/final")
    tokenizer.save_pretrained(f"{MODELS_DIR}/final")

    # Save predictions
    test_df_with_preds = test_df.copy()
    test_df_with_preds['distilbert_pred'] = test_preds
    test_df_with_preds.to_csv(f"{RESULTS_DIR}/transformer_predictions.csv", index=False)

    # Save metrics
    results = {
        'model': 'distilbert-base-uncased',
        'dev': dev_metrics,
        'test': test_metrics,
        'fold_metrics': fold_metrics
    }

    with open(f"{RESULTS_DIR}/transformer_metrics.json", 'w') as f:
        # Convert metrics to serializable format
        serializable_results = {
            'model': results['model'],
            'dev': results['dev'],
            'test': results['test'],
            'fold_metrics': []
        }
        for m in fold_metrics:
            serializable_results['fold_metrics'].append({
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in m.items() if 'f1' in k.lower() or 'accuracy' in k.lower()
            })
        json.dump(serializable_results, f, indent=2)

    print("\n" + "=" * 50)
    print(f"[DONE] Model saved to {MODELS_DIR}/final")
    print(f"       Predictions saved to {RESULTS_DIR}/transformer_predictions.csv")
    print(f"       Metrics saved to {RESULTS_DIR}/transformer_metrics.json")


if __name__ == "__main__":
    main()
