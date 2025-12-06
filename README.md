# Hedge Detection in Financial News

An NLP project for detecting hedging language in financial news articles. Hedging refers to linguistic expressions that convey uncertainty, speculation, or tentative claims (e.g., "may increase", "could potentially", "analysts expect").

## Project Overview

This project implements and compares multiple approaches to hedge detection:

1. **Baseline Model**: Rule-based keyword matching using predefined hedge cues
2. **Classical ML Models**: Logistic Regression and Linear SVM with TF-IDF features
3. **Transformer Models**: Fine-tuned DistilBERT and FinBERT for sequence classification

## Repository Structure

```
├── data/
│   ├── annotations/          # Annotated sentence data
│   ├── external/             # FNSPID dataset
│   └── splits/               # Train/dev/test splits
├── models/                   # Saved model checkpoints
│   ├── logistic_regression.joblib
│   ├── svm.joblib
│   └── tfidf_vectorizer.joblib
├── results/                  # Predictions and metrics
├── scripts/
│   ├── baseline_model.py         # Keyword matching baseline
│   ├── classical_ml_model.py     # Logistic Regression & SVM
│   ├── transformer_model.py      # DistilBERT fine-tuning
│   ├── finbert_model.py          # FinBERT fine-tuning (large dataset)
│   ├── prepare_training_data.py  # Create large training set from FNSPID
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
- transformers, torch
- joblib

## Models

### Available Models

| Model | Location | Size | Test F1 |
|-------|----------|------|---------|
| Baseline (Keywords) | Code only | - | ~0.50 |
| Logistic Regression | `models/logistic_regression.joblib` | 3 KB | ~0.60 |
| Linear SVM | `models/svm.joblib` | 3 KB | ~0.60 |
| TF-IDF Vectorizer | `models/tfidf_vectorizer.joblib` | 11 KB | - |
| FinBERT | [HuggingFace Hub](https://huggingface.co/Shauryajain21/hedge-finbert) | 438 MB | 0.998 |

## Usage

### Using the FinBERT Model (Recommended)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model from Hugging Face Hub
model_name = "Shauryajain21/hedge-finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Predict on a sentence
sentence = "The company may increase its dividend next quarter."
inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

print(f"Hedge: {'Yes' if prediction == 1 else 'No'}")
```

### Using Classical ML Models

```python
import joblib

# Load models
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
lr_model = joblib.load("models/logistic_regression.joblib")
svm_model = joblib.load("models/svm.joblib")

# Predict on a sentence
sentence = "The company may increase its dividend next quarter."
features = vectorizer.transform([sentence])

lr_pred = lr_model.predict(features)[0]
svm_pred = svm_model.predict(features)[0]

print(f"Logistic Regression: {'Hedge' if lr_pred == 1 else 'Non-hedge'}")
print(f"SVM: {'Hedge' if svm_pred == 1 else 'Non-hedge'}")
```

### Using the Baseline Model

```python
import re

HEDGE_KEYWORDS = [
    'may', 'might', 'could', 'would', 'possibly', 'perhaps', 'likely', 'unlikely',
    'appear', 'seem', 'suggest', 'indicate', 'estimate', 'forecast', 'expect',
    'expected', 'project', 'potential', 'approximately', 'around', 'about',
    'believed', 'appears', 'seems', 'reported', 'reportedly', 'alleged',
    'allegedly', 'rumored', 'speculation', 'speculate', 'anticipated',
    'probably', 'presumably', 'poised', 'set to', 'plans to', 'aims to'
]

HEDGE_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(kw) for kw in HEDGE_KEYWORDS) + r')\b',
    re.IGNORECASE
)

def is_hedge(sentence):
    return bool(HEDGE_PATTERN.search(sentence))

# Example
sentence = "The company may increase its dividend next quarter."
print(f"Hedge: {is_hedge(sentence)}")
```

## Training Your Own Models

### 1. Prepare Training Data (for FinBERT)

```bash
python scripts/prepare_training_data.py
```

This creates train/val/test splits from the FNSPID dataset:
- Train: ~38,000 samples
- Val: ~4,800 samples
- Test: ~4,800 samples

### 2. Train Models

**Classical ML Models:**
```bash
python scripts/classical_ml_model.py
```

**FinBERT (requires GPU, ~90 min on Mac M-series):**
```bash
python scripts/finbert_model.py
```

### 3. Evaluate All Models

```bash
python scripts/evaluate.py
```

## Results

### Model Comparison (Test Set)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Baseline (Keywords) | 0.765 | 0.500 | 1.000 | 0.667 |
| Logistic Regression | 0.824 | 0.667 | 0.400 | 0.500 |
| Linear SVM | 0.824 | 0.667 | 0.400 | 0.500 |
| FinBERT (Large) | **0.999** | **0.997** | **0.998** | **0.998** |

*FinBERT significantly outperforms other models due to:*
- Financial domain pre-training
- Larger training dataset (~38K samples vs 64 samples)

### Results Files

- `results/baseline_metrics.json` - Baseline model performance
- `results/classical_ml_metrics.json` - LR and SVM performance
- `results/transformer_metrics.json` - DistilBERT performance
- `results/finbert_metrics.json` - FinBERT performance
- `results/comparison_report.json` - Side-by-side comparison

## Datasets

### Gold-labeled Data
- `data/splits/dev.csv` - 64 manually annotated sentences
- `data/splits/test.csv` - 17 manually annotated sentences

### FNSPID Dataset
- `data/external/fnspid_clean.csv` - Financial news articles
- Used to create larger training set with keyword-based pseudo-labels

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
| FinBERT | Contextual embeddings | 5 epochs, early stopping |

## License

MIT License
