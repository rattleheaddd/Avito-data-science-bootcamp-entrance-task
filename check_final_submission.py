import pandas as pd

df = pd.read_csv("submission.csv")

print("rows:", len(df))
print("columns:", df.columns.tolist())

print("NaN:", df.isna().sum())
print("duplicates:", df["image_id"].duplicated().sum())

print("min p:", df["p_180"].min())
print("max p:", df["p_180"].max())

print("below 0:", (df["p_180"] < 0).sum())
print("above 1:", (df["p_180"] > 1).sum())

sample = pd.read_csv("sample_submission.csv")
print(len(df), len(sample))
print((df["image_id"] == sample["image_id"]).all())