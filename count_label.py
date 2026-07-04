import pandas as pd
df = pd.read_csv("labeled_cv_data.csv", encoding="utf-8-sig")
print("--- Label Counts ---")
print(df['label'].value_counts())
