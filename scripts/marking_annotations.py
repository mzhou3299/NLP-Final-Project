import pandas as pd

annotate_df  = pd.read_csv('data/annotations/to_annotate.csv').copy()
for index, row in annotate_df.iterrows():
    print(row['sentence'])
    user_input = input("0(not hedged) or 1(hedged)")
    annotate_df.loc[index, 'gold_label'] = user_input

annotate_df.to_csv('data/annotations/annotation_complete.csv')