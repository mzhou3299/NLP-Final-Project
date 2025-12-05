# Hedge Detection in Financial News

An NLP project for detecting hedging language in financial news articles. Hedging refers to linguistic expressions that convey uncertainty, speculation, or tentative claims (e.g., "may increase", "could potentially", "analysts expect").

## Project Overview

This project implements and compares multiple approaches to hedge detection:

1. **Baseline Model**: Rule-based keyword matching using predefined hedge cues
2. **Classical ML Models**: Logistic Regression and Linear SVM with TF-IDF features
3. **Transformer Model**: Fine-tuned DistilBERT for sequence classification

## Repository Structure

```
├── data/
│   ├── annotations/          # Annotated sentence data
│   └── splits/               # Train/dev/test splits
├── models/                   # Saved model checkpoints
├── results/                  # Predictions and metrics
├── scripts/
│   ├── build_annotation_set.py   # Sample sentences for annotation
│   ├── split_data.py             # Create stratified data splits
│   ├── baseline_model.py         # Keyword matching baseline
│   ├── classical_ml_model.py     # Logistic Regression & SVM
│   ├── transformer_model.py      # DistilBERT fine-tuning
│   └── evaluate.py               # Model comparison
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- pandas, numpy, scikit-learn
- transformers, torch (for transformer model)

## Usage

### 1. Prepare Data Splits
```bash
python scripts/split_data.py
```
Creates stratified 80/20 dev/test splits with 5-fold cross-validation indices.

### 2. Run Models

**Baseline (Keyword Matching)**
```bash
python scripts/baseline_model.py
```

**Classical ML (Logistic Regression & SVM)**
```bash
python scripts/classical_ml_model.py
```

**Transformer (DistilBERT)**
```bash
python scripts/transformer_model.py
```

### 3. Compare Results
```bash
python scripts/evaluate.py
```
Generates a comprehensive comparison report of all models.

## Methodology

### Hedge Cues
The following linguistic markers are used to identify potential hedges:
- Modal verbs: *may*, *might*, *could*
- Probability terms: *likely*, *unlikely*, *possibly*, *probable*
- Expectation verbs: *expects*, *forecast*, *anticipate*, *project*
- Epistemic markers: *suggests*, *appears*, *seems*, *believe*, *estimates*
- Uncertainty indicators: *uncertain*, *potential*, *risk*

### Model Details

| Model | Features | Training |
|-------|----------|----------|
| Baseline | Keyword matching | None (rule-based) |
| Logistic Regression | TF-IDF (unigrams + bigrams) | GridSearchCV, 5-fold CV |
| Linear SVM | TF-IDF (unigrams + bigrams) | GridSearchCV, 5-fold CV |
| DistilBERT | Contextual embeddings | 5-fold CV, early stopping |

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Results

Results are saved to the `results/` directory:
- `baseline_metrics.json` - Baseline model performance
- `classical_ml_metrics.json` - LR and SVM performance
- `transformer_metrics.json` - DistilBERT performance
- `comparison_report.json` - Side-by-side comparison

## License

MIT License
