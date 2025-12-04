"""
evaluate.py - Unified evaluation script comparing all hedge detection models.

Loads metrics from all models and generates a comparison report.
"""

import json
from pathlib import Path

RESULTS_DIR = "results"


def load_metrics(filename):
    """Load metrics from a JSON file."""
    filepath = Path(RESULTS_DIR) / filename
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


def print_separator(char="=", length=70):
    print(char * length)


def print_model_results(name, metrics, dataset="test"):
    """Print formatted results for a single model."""
    if metrics is None:
        print(f"\n{name}: No results found")
        return

    data = metrics.get(dataset, metrics)

    print(f"\n{name}")
    print("-" * 40)
    print(f"  Accuracy:  {data['accuracy']:.3f}")
    print(f"  Precision: {data['precision']:.3f}")
    print(f"  Recall:    {data['recall']:.3f}")
    print(f"  F1 Score:  {data['f1']:.3f}")

    if 'tp' in data:
        print(f"\n  Confusion Matrix:")
        print(f"            Pred 0  Pred 1")
        print(f"  Actual 0    {data['tn']:3d}     {data['fp']:3d}")
        print(f"  Actual 1    {data['fn']:3d}     {data['tp']:3d}")


def main():
    print_separator()
    print("HEDGE DETECTION MODEL COMPARISON")
    print_separator()

    # Load all metrics
    baseline_metrics = load_metrics("baseline_metrics.json")
    classical_metrics = load_metrics("classical_ml_metrics.json")
    transformer_metrics = load_metrics("transformer_metrics.json")

    # Print individual results
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)

    print_model_results("BASELINE (Keyword Matching)", baseline_metrics)

    if classical_metrics:
        print_model_results("LOGISTIC REGRESSION", classical_metrics.get('logistic_regression'))
        print_model_results("LINEAR SVM", classical_metrics.get('svm'))

    print_model_results("DISTILBERT", transformer_metrics)

    # Summary comparison table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY (Test Set)")
    print("=" * 70)

    print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "Model", "Accuracy", "Precision", "Recall", "F1"
    ))
    print("-" * 70)

    models = []

    if baseline_metrics:
        models.append(("Baseline (Keyword)", baseline_metrics.get('test', baseline_metrics)))

    if classical_metrics:
        if 'logistic_regression' in classical_metrics:
            models.append(("Logistic Regression", classical_metrics['logistic_regression'].get('test')))
        if 'svm' in classical_metrics:
            models.append(("Linear SVM", classical_metrics['svm'].get('test')))

    if transformer_metrics:
        models.append(("DistilBERT", transformer_metrics.get('test')))

    for name, m in models:
        if m:
            print("{:<25} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
                name, m['accuracy'], m['precision'], m['recall'], m['f1']
            ))

    # Find best model
    print("\n" + "-" * 70)
    if models:
        best_f1 = max(models, key=lambda x: x[1]['f1'] if x[1] else 0)
        best_acc = max(models, key=lambda x: x[1]['accuracy'] if x[1] else 0)
        print(f"Best F1 Score:  {best_f1[0]} ({best_f1[1]['f1']:.3f})")
        print(f"Best Accuracy:  {best_acc[0]} ({best_acc[1]['accuracy']:.3f})")

    # Dev set comparison (if available)
    print("\n" + "=" * 70)
    print("DEV SET RESULTS (Cross-Validation)")
    print("=" * 70)

    dev_models = []

    if baseline_metrics and 'dev' in baseline_metrics:
        dev_models.append(("Baseline (Keyword)", baseline_metrics['dev']))

    if classical_metrics:
        if 'logistic_regression' in classical_metrics and 'dev' in classical_metrics['logistic_regression']:
            dev_models.append(("Logistic Regression", classical_metrics['logistic_regression']['dev']))
        if 'svm' in classical_metrics and 'dev' in classical_metrics['svm']:
            dev_models.append(("Linear SVM", classical_metrics['svm']['dev']))

    if transformer_metrics and 'dev' in transformer_metrics:
        dev_models.append(("DistilBERT", transformer_metrics['dev']))

    if dev_models:
        print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
            "Model", "Accuracy", "Precision", "Recall", "F1"
        ))
        print("-" * 70)

        for name, m in dev_models:
            if m:
                print("{:<25} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
                    name, m['accuracy'], m['precision'], m['recall'], m['f1']
                ))

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("""
Key Observations:

1. BASELINE (Keyword Matching):
   - High recall (captures most hedges)
   - Lower precision (many false positives)
   - Simple but effective first-pass filter

2. CLASSICAL ML (Logistic Regression / SVM):
   - Better precision-recall trade-off
   - Benefits from TF-IDF feature representation
   - SVM often outperforms LR on small datasets

3. TRANSFORMER (DistilBERT):
   - Limited by small training set (64 samples)
   - Prone to overfitting with few examples
   - Would likely improve with more data or data augmentation

Recommendations:
- For production: Use SVM or ensemble of baseline + ML
- For better transformer results: Collect more training data
- Consider domain-specific pre-training (FinBERT) with more data
""")

    # Save comparison report
    report = {
        'test_results': {name: m for name, m in models},
        'dev_results': {name: m for name, m in dev_models},
        'best_f1_model': best_f1[0] if models else None,
        'best_accuracy_model': best_acc[0] if models else None
    }

    with open(f"{RESULTS_DIR}/comparison_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n[DONE] Full report saved to {RESULTS_DIR}/comparison_report.json")


if __name__ == "__main__":
    main()
