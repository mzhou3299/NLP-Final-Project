"""
evaluate_all_models.py

Script to evaluate baseline / classical ML / transformer hedge-detection models on dev and test splits.
Outputs evaluation metrics summary and error analysis CSVs for false positives/negatives.

Usage:
    python scripts/evaluate_all_models.py

Results will be saved in:
    - results/evaluation_summary.csv
    - results/false_positives_*.csv
    - results/false_negatives_*.csv
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import joblib
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Add current directory to path to allow importing baseline_model
sys.path.append(str(Path(__file__).parent))

try:
    import baseline_model
except ImportError:
    # Attempt to import if running from root
    try:
        from scripts import baseline_model
    except ImportError:
        print("Warning: Could not import baseline_model. Baseline evaluation will be skipped.")
        baseline_model = None

# Configuration
DEV_PATH = "data/splits/dev.csv"
TEST_PATH = "data/splits/test.csv"
RESULTS_DIR = "results"
MODELS_DIR = "models"
FINBERT_MODEL_PATH = "models/finbert/final"

def load_data(split_path: str) -> pd.DataFrame:
    """Load data split (dev or test)."""
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Data file not found: {split_path}")
    
    df = pd.read_csv(split_path)
    # Ensure columns exist
    if 'sentence' not in df.columns or 'gold_label' not in df.columns:
        raise ValueError(f"File {split_path} must contain 'sentence' and 'gold_label' columns.")
    
    return df

def calculate_metrics(y_true, y_pred, y_probs=None):
    """Calculate evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_macro': f1_macro,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }

    if y_probs is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            metrics['auc'] = auc(fpr, tpr)
        except Exception as e:
            print(f"Warning: Could not compute AUC: {e}")
            metrics['auc'] = None

    return metrics

def save_error_analysis(df, y_true, y_pred, model_name, dataset_name, y_probs=None):
    """Save false positives and false negatives to CSV."""
    results_df = df.copy()
    results_df['prediction'] = y_pred
    results_df['gold_label'] = y_true # ensure using the aligned truth
    if y_probs is not None:
        results_df['probability'] = y_probs

    # False Positives: Predicted 1, Gold 0
    fp_df = results_df[(results_df['prediction'] == 1) & (results_df['gold_label'] == 0)]
    fp_path = Path(RESULTS_DIR) / f"false_positives_{model_name}_{dataset_name}.csv"
    fp_df.to_csv(fp_path, index=False)

    # False Negatives: Predicted 0, Gold 1
    fn_df = results_df[(results_df['prediction'] == 0) & (results_df['gold_label'] == 1)]
    fn_path = Path(RESULTS_DIR) / f"false_negatives_{model_name}_{dataset_name}.csv"
    fn_df.to_csv(fn_path, index=False)
    
    print(f"  Saved error analysis to {RESULTS_DIR}/")

def evaluate_baseline(df, dataset_name, summary_list):
    """Evaluate baseline keyword model."""
    if baseline_model is None:
        return

    print(f"\nEvaluating Baseline Model on {dataset_name}...")
    sentences = df['sentence'].tolist()
    y_true = df['gold_label'].tolist()
    
    # Predict
    y_pred = [baseline_model.predict_hedge(s) for s in sentences]
    
    # Metrics
    metrics = calculate_metrics(y_true, y_pred)
    metrics['model'] = 'baseline'
    metrics['dataset'] = dataset_name
    summary_list.append(metrics)
    
    print(f"  Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
    
    # Error Analysis
    save_error_analysis(df, y_true, y_pred, 'baseline', dataset_name)

def evaluate_classical(df, dataset_name, summary_list):
    """Evaluate Logistic Regression and SVM."""
    print(f"\nEvaluating Classical ML Models on {dataset_name}...")
    
    # Load Vectorizer
    vect_path = Path(MODELS_DIR) / "tfidf_vectorizer.joblib"
    if not vect_path.exists():
        print(f"  Vectorizer not found at {vect_path}. Skipping.")
        return

    vectorizer = joblib.load(vect_path)
    X_tfidf = vectorizer.transform(df['sentence'].tolist())
    y_true = df['gold_label'].values
    
    models_to_eval = [
        ('logistic_regression', 'logistic_regression.joblib'),
        ('svm', 'svm.joblib')
    ]
    
    for model_name, filename in models_to_eval:
        model_path = Path(MODELS_DIR) / filename
        if not model_path.exists():
            print(f"  Model {model_name} not found at {model_path}. Skipping.")
            continue
            
        print(f"  Running {model_name}...")
        model = joblib.load(model_path)
        y_pred = model.predict(X_tfidf)
        
        # Get probabilities if available
        y_probs = None
        if hasattr(model, "predict_proba"):
            try:
                y_probs = model.predict_proba(X_tfidf)[:, 1]
            except:
                pass
        elif hasattr(model, "decision_function"):
             # For SVM, decision_function gives distance, not prob. 
             # We can use it as a score for ROC but it's not strictly 0-1 prob without calibration.
             # We'll skip AUC for SVM unless it has predict_proba (requires probability=True in SVC, LinearSVC doesn't support it directly)
             pass

        metrics = calculate_metrics(y_true, y_pred, y_probs)
        metrics['model'] = model_name
        metrics['dataset'] = dataset_name
        summary_list.append(metrics)
        
        print(f"    Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
        save_error_analysis(df, y_true, y_pred, model_name, dataset_name, y_probs)

def evaluate_finbert(df, dataset_name, summary_list):
    """Evaluate FinBERT model."""
    print(f"\nEvaluating FinBERT Model on {dataset_name}...")
    
    if not os.path.exists(FINBERT_MODEL_PATH):
        print(f"  FinBERT model not found at {FINBERT_MODEL_PATH}. Skipping.")
        return

    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_PATH)
    except Exception as e:
        print(f"  Error loading FinBERT: {e}")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    sentences = df['sentence'].tolist()
    y_true = df['gold_label'].values
    
    predictions = []
    probabilities = []
    
    batch_size = 16
    for i in tqdm(range(0, len(sentences), batch_size), desc="  Predicting"):
        batch_sentences = sentences[i:i+batch_size]
        
        inputs = tokenizer(
            batch_sentences, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())
            
    y_pred = np.array(predictions)
    y_probs = np.array(probabilities)
    
    metrics = calculate_metrics(y_true, y_pred, y_probs)
    metrics['model'] = 'finbert'
    metrics['dataset'] = dataset_name
    summary_list.append(metrics)
    
    print(f"  Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
    save_error_analysis(df, y_true, y_pred, 'finbert', dataset_name, y_probs)

def main():
    # Ensure results directory exists
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    summary_list = []
    
    # Process both splits
    for split_name, split_path in [('dev', DEV_PATH), ('test', TEST_PATH)]:
        if not os.path.exists(split_path):
            print(f"Skipping {split_name}: File not found.")
            continue
            
        print(f"\n{'='*20} Processing {split_name.upper()} split {'='*20}")
        df = load_data(split_path)
        print(f"Loaded {len(df)} samples.")
        
        evaluate_baseline(df, split_name, summary_list)
        evaluate_classical(df, split_name, summary_list)
        evaluate_finbert(df, split_name, summary_list)

    # Save summary
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        summary_path = Path(RESULTS_DIR) / "evaluation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'-'*50}")
        print(f"Evaluation Complete. Summary saved to {summary_path}")
        print(f"Full details in {RESULTS_DIR}/")
    else:
        print("\nNo evaluation results produced.")

if __name__ == "__main__":
    main()


