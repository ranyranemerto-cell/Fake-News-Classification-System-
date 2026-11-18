# ============================================================
# FAKE NEWS CLASSIFICATION SYSTEM
# Model Training and Evaluation
# ============================================================
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# 1. LOAD CLEANED DATASET
# ------------------------------------------------------------

print("\n======================================")
print("     FAKE NEWS MODEL TRAINING")
print("======================================")

df = pd.read_csv("WELFake_Cleaned.csv")

print("\nDataset loaded")
print("Dataset shape:", df.shape)

required_columns = ["title", "text", "label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# ------------------------------------------------------------
# 2. PREPARE INPUT FEATURES AND TARGET
# ------------------------------------------------------------

df["content"] = (
    df["title"].fillna("").astype(str)
    + " "
    + df["text"].fillna("").astype(str)
)

X = df["content"]

y = df["label"]

print("\nInput features: News Title and Text")
print("Target variable: label")
print("\nLabel distribution:")
print(y.value_counts())

# ------------------------------------------------------------
# 3. SPLIT DATASET
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset successfully split!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ------------------------------------------------------------
# 4. TF-IDF VECTORIZATION
# ------------------------------------------------------------

print("\nApplying TF-IDF Vectorization...")

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF vectorization completed!")
print("Training matrix:", X_train_tfidf.shape)
print("Testing matrix:", X_test_tfidf.shape)

# ------------------------------------------------------------
# NEAR-DUPLICATE CHECK (train vs test)
# ------------------------------------------------------------

print("\nChecking near-duplicates between train and test...")

rng = np.random.RandomState(42)
idx = rng.choice(X_test_tfidf.shape[0], size=2000, replace=False)
sample = X_test_tfidf[idx]

max_sims = []
for i in range(0, sample.shape[0], 250):
    sims = (sample[i:i + 250] @ X_train_tfidf.T).toarray()
    max_sims.extend(sims.max(axis=1))
max_sims = np.array(max_sims)

for thr in (0.80, 0.90, 0.95):
    pct = (max_sims >= thr).mean() * 100
    print(f"Test articles with similarity >= {thr:.2f} to a training article: {pct:.2f}%")

# ------------------------------------------------------------
# 5. INITIALIZE MACHINE LEARNING MODELS
# ------------------------------------------------------------

models = {
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}

results = []
trained_models = {}

# ------------------------------------------------------------
# 6. TRAIN AND EVALUATE MODELS
# ------------------------------------------------------------

for model_name, model in models.items():

    print("\n======================================")
    print("Training:", model_name)
    print("======================================")

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    print("\nModel Performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["REAL", "FAKE"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    trained_models[model_name] = model

# ------------------------------------------------------------
# 7. COMPARE MODEL PERFORMANCE
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n======================================")
print("       MODEL COMPARISON")
print("======================================")

print(results_df.to_string(index=False))

# ------------------------------------------------------------
# 8. SELECT BEST MODEL
# ------------------------------------------------------------

best_model_name = results_df.sort_values(
    by="F1-Score",
    ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)

# ------------------------------------------------------------
# 9. SAVE BEST MODEL AND TF-IDF VECTORIZER
# ------------------------------------------------------------

joblib.dump(best_model, "fake_news_model.joblib")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")

print("\n======================================")
print("       MODEL SAVING COMPLETED")
print("======================================")

print("Saved model: fake_news_model.joblib")
print("Saved vectorizer: tfidf_vectorizer.joblib")

print("\nModel training and evaluation completed.")

# ------------------------------------------------------------
# SAVE METRICS FOR THE APP
# ------------------------------------------------------------

y_pred_best = best_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred_best)

metrics = {
    "best_model": best_model_name,
    "comparison": results_df.round(4).to_dict(orient="records"),
    "confusion_matrix": cm.tolist(),
    "train_size": len(X_train),
    "test_size": len(X_test),
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved metrics: metrics.json")