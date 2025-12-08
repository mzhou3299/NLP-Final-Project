import pandas as pd
from sklearn.metrics import cohen_kappa_score

# load both files
df1 = pd.read_csv("data/annotations/gold_label_set_1.csv")
df2 = pd.read_csv("data/annotations/gold_label_set_2.csv")

kappa = cohen_kappa_score(df1["gold_label"], df2["gold_label"])

print("Cohen's kappa:", kappa)
