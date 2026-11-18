import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("WELFake_Dataset.csv")

print("=" * 50)
print("       WELFAKE DATASET VERIFICATION")
print("=" * 50)

# ==========================================
# 2. BASIC DATASET INFORMATION
# ==========================================

print("\n--- BASIC DATASET INFO ---")
print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# ==========================================
# 3. DATA QUALITY CHECKS
# ==========================================

print("\n--- DATA QUALITY CHECKS ---")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFully Duplicated Rows:", df.duplicated().sum())

# ==========================================
# 4. REMOVE MISSING AND EMPTY VALUES
# ==========================================

df_cleaned = df.dropna(
    subset=["title", "text", "label"]
).copy()

# Remove leading and trailing spaces
df_cleaned["title"] = df_cleaned["title"].astype(str).str.strip()
df_cleaned["text"] = df_cleaned["text"].astype(str).str.strip()

# Remove empty titles and texts
df_cleaned = df_cleaned[
    (df_cleaned["title"] != "") &
    (df_cleaned["text"] != "")
].copy()

print("\nRecords after removing missing/empty values:", len(df_cleaned))

# ==========================================
# 5. CHECK DUPLICATE TEXT CONTENT
# ==========================================

print("\n--- DUPLICATE TEXT CHECK ---")

duplicate_texts = df_cleaned["text"].duplicated().sum()

print("Duplicate Text Content:", duplicate_texts)

# ==========================================
# 6. CHECK CONFLICTING LABELS
# ==========================================

conflicting_labels = (
    df_cleaned.groupby("text")["label"].nunique() > 1
).sum()

print("Texts with Conflicting Labels:", conflicting_labels)

# ==========================================
# 7. REMOVE CONFLICTING LABELS AND DUPLICATES
# ==========================================

label_counts = df_cleaned.groupby("text")["label"].nunique()
conflicting_texts = label_counts[label_counts > 1].index
df_cleaned = df_cleaned[~df_cleaned["text"].isin(conflicting_texts)]

df_cleaned = df_cleaned.drop_duplicates(
    subset=["text"],
    keep="first"
).copy()

# ==========================================
# 8. FINAL DATASET VERIFICATION
# ==========================================

print("\n" + "=" * 50)
print("          FINAL CLEANED DATASET")
print("=" * 50)

print("Original Records:", len(df))
print("Final Valid Original Records:", len(df_cleaned))
print("Final Dataset Shape:", df_cleaned.shape)

# ==========================================
# 9. LABEL DISTRIBUTION
# ==========================================

print("\n--- LABEL DISTRIBUTION ---")

print(df_cleaned["label"].value_counts())

print("\nLabel Percentages:")
print(
    (df_cleaned["label"].value_counts(normalize=True) * 100)
    .round(2)
)

# ==========================================
# 10. FINAL DATA QUALITY CHECK
# ==========================================

print("\n--- FINAL DATA QUALITY CHECK ---")

print("Remaining Missing Values:")
print(df_cleaned[["title", "text", "label"]].isnull().sum())

print(
    "Remaining Duplicate Texts:",
    df_cleaned["text"].duplicated().sum()
)

# ==========================================
# 11. SAVE CLEANED DATASET
# ==========================================

df_cleaned.to_csv(
    "WELFake_Cleaned.csv",
    index=False
)

print("\nCleaned dataset saved as WELFake_Cleaned.csv")

print("\nDataset verification completed.")
print("=" * 50)
