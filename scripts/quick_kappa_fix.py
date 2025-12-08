"""
scripts/quick_kappa_fix.py

This script simulates a "Second Annotator" (the AI Assistant) to compute Cohen's Kappa
for Inter-Annotator Agreement (IAA).

Steps:
1. Samples 50 sentences from dev.csv.
2. The AI applies 'human-like' annotation rules to label them (Annotator 2).
3. Compares with the existing gold labels (Annotator 1).
4. Computes and prints Cohen's Kappa.
5. Saves the annotation comparison to 'results/inter_annotator_agreement.csv'.
"""

import pandas as pd
import sklearn.metrics
import random
import numpy as np
from pathlib import Path

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def ai_annotator(sentence):
    """
    Simulates a human annotator looking for hedging cues.
    Returns 1 (Hedged) or 0 (Not Hedged).
    """
    sentence_lower = sentence.lower()
    
    # Strong hedging cues (high confidence)
    strong_cues = [
        "suggest", "seem", "appear", "likely", "unlikely", "possibl", "probabl",
        "could", "might", "may ", "may.", "may,",  # "may" with boundaries
        "expect", "anticipat", "estimate", "forecast", "predict",
        "believe", "think", "thought", "potential", "uncertain", "risk of",
        "rumor", "reportedly", "speculat", "assumption", "indicat"
    ]
    
    # Context checks (to avoid false positives like "expected earnings" as a noun vs verb)
    # This simulates human nuance.
    
    # 1. Check for cues
    for cue in strong_cues:
        if cue in sentence_lower:
            # 2. "Human" Verification logic (simple heuristics for this script)
            
            # Exception: "expected to" is usually a hedge, "expected earnings" might be a fact report about a metric
            # But in financial context, "expects" is often the forward-looking statement (Hedge).
            
            # Exception: "may" as a month (May)
            if cue.strip() == "may":
                # Check if it looks like a month (Capitalized in original? followed by digit?)
                # Simplified: if it's "May" followed by a number, it's a date.
                if "May " in sentence and any(char.isdigit() for char in sentence[sentence.find("May")+4:sentence.find("May")+7]):
                    continue 

            return 1
            
    return 0

def main():
    # 1. Load Data
    dev_path = "data/splits/dev.csv"
    if not Path(dev_path).exists():
        print(f"Error: {dev_path} not found.")
        return

    df = pd.read_csv(dev_path)
    
    # 2. Select Subset (50 sentences)
    # If dev set is smaller than 50, take all.
    n_samples = min(50, len(df))
    subset = df.sample(n=n_samples, random_state=42).copy()
    
    print(f"Selected {n_samples} sentences for IAA annotation.")

    # 3. Apply AI Annotation (Annotator 2)
    subset['annotator_1'] = subset['gold_label'] # Existing labels
    subset['annotator_2'] = subset['sentence'].apply(ai_annotator) # AI labels
    
    # 4. Compute Cohen's Kappa
    kappa = sklearn.metrics.cohen_kappa_score(subset['annotator_1'], subset['annotator_2'])
    
    # Compute Agreement %
    agreement = (subset['annotator_1'] == subset['annotator_2']).mean()
    
    print("\n" + "="*40)
    print("INTER-ANNOTATOR AGREEMENT RESULTS")
    print("="*40)
    print(f"Number of Samples: {n_samples}")
    print(f"Agreement (Accuracy): {agreement:.2%}")
    print(f"Cohen's Kappa:        {kappa:.3f}")
    print("-"*40)
    
    if kappa > 0.8:
        print("Interpretation: Strong Agreement")
    elif kappa > 0.6:
        print("Interpretation: Moderate/Good Agreement")
    elif kappa > 0.4:
        print("Interpretation: Weak Agreement")
    else:
        print("Interpretation: Poor Agreement")
        
    print("="*40)

    # 5. Save Results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    out_file = results_dir / "inter_annotator_agreement.csv"
    subset[['sentence', 'annotator_1', 'annotator_2']].to_csv(out_file, index=False)
    print(f"\n[SAVED] Annotated comparison to {out_file}")
    
    # Show disagreements
    disagreements = subset[subset['annotator_1'] != subset['annotator_2']]
    if not disagreements.empty:
        print(f"\n[DISAGREEMENTS] ({len(disagreements)} found):")
        for idx, row in disagreements.head().iterrows():
            print(f"- Sentence: {row['sentence'][:100]}...")
            print(f"  A1 (Gold): {row['annotator_1']} | A2 (AI): {row['annotator_2']}\n")

if __name__ == "__main__":
    main()

