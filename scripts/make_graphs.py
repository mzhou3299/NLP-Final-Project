"""
scripts/make_graphs.py

Generates visualizations for the NLP Final Project presentation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})
RESULTS_DIR = Path("results")
GRAPHS_DIR = RESULTS_DIR / "graphs"
GRAPHS_DIR.mkdir(exist_ok=True)

def plot_performance_comparison():
    """Bar chart comparing F1 and Accuracy across models."""
    csv_path = RESULTS_DIR / "evaluation_summary.csv"
    if not csv_path.exists():
        print("Summary CSV not found.")
        return

    df = pd.read_csv(csv_path)
    # Filter for Test set only for the main comparison
    df_test = df[df['dataset'] == 'test'].copy()
    
    # Melt for seaborn
    df_melt = df_test.melt(
        id_vars=['model'], 
        value_vars=['accuracy', 'f1', 'precision', 'recall'],
        var_name='Metric', 
        value_name='Score'
    )
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_melt, x='model', y='Score', hue='Metric', palette="viridis")
    
    plt.title('Model Performance on Test Set', fontsize=16)
    plt.ylabel('Score (0-1)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

    plt.tight_layout()
    save_path = GRAPHS_DIR / "model_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")

def plot_confusion_matrices():
    """Generate heatmaps for confusion matrices."""
    csv_path = RESULTS_DIR / "evaluation_summary.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    df_test = df[df['dataset'] == 'test']

    for _, row in df_test.iterrows():
        model = row['model']
        tn, fp, fn, tp = row['tn'], row['fp'], row['fn'], row['tp']
        
        # Construct matrix
        matrix = [[tn, fp], [fn, tp]]
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred No', 'Pred Yes'],
                    yticklabels=['Actual No', 'Actual Yes'])
        
        plt.title(f'Confusion Matrix: {model.title()}', fontsize=14)
        plt.tight_layout()
        
        save_path = GRAPHS_DIR / f"cm_{model}.png"
        plt.savefig(save_path, dpi=300)
        print(f"Saved {save_path}")

def plot_kappa_agreement():
    """Visualize Inter-Annotator Agreement."""
    csv_path = RESULTS_DIR / "inter_annotator_agreement.csv"
    if not csv_path.exists():
        return
        
    df = pd.read_csv(csv_path)
    
    # Categories
    df['Agreement'] = df.apply(
        lambda x: 'Agreed' if x['annotator_1'] == x['annotator_2'] else 
                  ('Missed Hedge (A2 found)' if x['annotator_2'] == 1 else 'False Alarm (A2 found)'),
        axis=1
    )
    
    counts = df['Agreement'].value_counts()
    
    plt.figure(figsize=(8, 6))
    colors = sns.color_palette('pastel')[0:3]
    
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Inter-Annotator Agreement (50 Samples)', fontsize=16)
    
    save_path = GRAPHS_DIR / "annotator_agreement.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")

def main():
    print("Generating graphs...")
    plot_performance_comparison()
    plot_confusion_matrices()
    plot_kappa_agreement()
    print("Done!")

if __name__ == "__main__":
    main()

