"""
classical_ml_model.py - Logistic Regression and SVM with TF-IDF features.

Uses 5-fold stratified cross-validation for hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DEV_PATH = "data/splits/dev.csv"
TEST_PATH = "data/splits/test.csv"
MODELS_DIR = "models"
RESULTS_DIR = "results"


def evaluate(y_true, y_pred):
    """Calculate accuracy, precision, recall, F1."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

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


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with GridSearchCV."""
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'class_weight': ['balanced', None]
    }

    lr = LogisticRegression(max_iter=1000, random_state=42)
    grid = GridSearchCV(lr, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV F1:  {grid.best_score_:.3f}")

    return grid.best_estimator_


def train_svm(X_train, y_train):
    """Train Linear SVM with GridSearchCV."""
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'class_weight': ['balanced', None]
    }

    svm = LinearSVC(max_iter=2000, random_state=42, dual=True)
    grid = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV F1:  {grid.best_score_:.3f}")

    return grid.best_estimator_


def main():
    # Load data
    dev_df = pd.read_csv(DEV_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_dev = dev_df['sentence'].tolist()
    y_dev = dev_df['gold_label'].values
    X_test = test_df['sentence'].tolist()
    y_test = test_df['gold_label'].values

    print("=" * 50)
    print("CLASSICAL ML MODELS: Logistic Regression & SVM")
    print("=" * 50)

    # TF-IDF vectorization
    print("\n[VECTORIZING] TF-IDF with unigrams + bigrams...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
        min_df=2,
        max_df=0.95
    )

    X_dev_tfidf = vectorizer.fit_transform(X_dev)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Feature matrix:  {X_dev_tfidf.shape}")

    # Create directories
    Path(MODELS_DIR).mkdir(exist_ok=True, parents=True)
    Path(RESULTS_DIR).mkdir(exist_ok=True, parents=True)

    # Save vectorizer
    joblib.dump(vectorizer, f"{MODELS_DIR}/tfidf_vectorizer.joblib")

    results = {}

    # =====================
    # LOGISTIC REGRESSION
    # =====================
    print("\n" + "-" * 50)
    print("LOGISTIC REGRESSION")
    print("-" * 50)

    print("\n[TRAINING] GridSearchCV with 5-fold CV...")
    lr_model = train_logistic_regression(X_dev_tfidf, y_dev)

    # Cross-validation predictions on dev set
    lr_dev_preds = cross_val_predict(lr_model, X_dev_tfidf, y_dev, cv=5)
    lr_dev_metrics = evaluate(y_dev, lr_dev_preds)

    print(f"\n[DEV SET - CV] ({len(y_dev)} samples)")
    print(f"  Accuracy:  {lr_dev_metrics['accuracy']:.3f}")
    print(f"  Precision: {lr_dev_metrics['precision']:.3f}")
    print(f"  Recall:    {lr_dev_metrics['recall']:.3f}")
    print(f"  F1 Score:  {lr_dev_metrics['f1']:.3f}")

    # Test set predictions
    lr_test_preds = lr_model.predict(X_test_tfidf)
    lr_test_metrics = evaluate(y_test, lr_test_preds)

    print(f"\n[TEST SET] ({len(y_test)} samples)")
    print(f"  Accuracy:  {lr_test_metrics['accuracy']:.3f}")
    print(f"  Precision: {lr_test_metrics['precision']:.3f}")
    print(f"  Recall:    {lr_test_metrics['recall']:.3f}")
    print(f"  F1 Score:  {lr_test_metrics['f1']:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"            Pred 0  Pred 1")
    print(f"  Actual 0    {lr_test_metrics['tn']:3d}     {lr_test_metrics['fp']:3d}")
    print(f"  Actual 1    {lr_test_metrics['fn']:3d}     {lr_test_metrics['tp']:3d}")

    # Save LR model and results
    joblib.dump(lr_model, f"{MODELS_DIR}/logistic_regression.joblib")
    results['logistic_regression'] = {
        'dev': lr_dev_metrics,
        'test': lr_test_metrics,
        'best_params': lr_model.get_params()
    }

    # =====================
    # SVM
    # =====================
    print("\n" + "-" * 50)
    print("LINEAR SVM")
    print("-" * 50)

    print("\n[TRAINING] GridSearchCV with 5-fold CV...")
    svm_model = train_svm(X_dev_tfidf, y_dev)

    # Cross-validation predictions on dev set
    svm_dev_preds = cross_val_predict(svm_model, X_dev_tfidf, y_dev, cv=5)
    svm_dev_metrics = evaluate(y_dev, svm_dev_preds)

    print(f"\n[DEV SET - CV] ({len(y_dev)} samples)")
    print(f"  Accuracy:  {svm_dev_metrics['accuracy']:.3f}")
    print(f"  Precision: {svm_dev_metrics['precision']:.3f}")
    print(f"  Recall:    {svm_dev_metrics['recall']:.3f}")
    print(f"  F1 Score:  {svm_dev_metrics['f1']:.3f}")

    # Test set predictions
    svm_test_preds = svm_model.predict(X_test_tfidf)
    svm_test_metrics = evaluate(y_test, svm_test_preds)

    print(f"\n[TEST SET] ({len(y_test)} samples)")
    print(f"  Accuracy:  {svm_test_metrics['accuracy']:.3f}")
    print(f"  Precision: {svm_test_metrics['precision']:.3f}")
    print(f"  Recall:    {svm_test_metrics['recall']:.3f}")
    print(f"  F1 Score:  {svm_test_metrics['f1']:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"            Pred 0  Pred 1")
    print(f"  Actual 0    {svm_test_metrics['tn']:3d}     {svm_test_metrics['fp']:3d}")
    print(f"  Actual 1    {svm_test_metrics['fn']:3d}     {svm_test_metrics['tp']:3d}")

    # Save SVM model and results
    joblib.dump(svm_model, f"{MODELS_DIR}/svm.joblib")
    results['svm'] = {
        'dev': svm_dev_metrics,
        'test': svm_test_metrics
    }

    # Save all predictions
    test_df_with_preds = test_df.copy()
    test_df_with_preds['lr_pred'] = lr_test_preds
    test_df_with_preds['svm_pred'] = svm_test_preds
    test_df_with_preds.to_csv(f"{RESULTS_DIR}/classical_ml_predictions.csv", index=False)

    # Save metrics
    with open(f"{RESULTS_DIR}/classical_ml_metrics.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 50)
    print("[DONE] Models saved to models/")
    print("       Predictions saved to results/classical_ml_predictions.csv")
    print("       Metrics saved to results/classical_ml_metrics.json")


if __name__ == "__main__":
    main()
