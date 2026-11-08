
from pathlib import Path

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Project directories
DATA_DIR = Path("data/raw/liar")
MODEL_DIR = Path("reports/models")
RESULT_DIR = Path("reports/baseline")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "job_title", "state_info", "party_affiliation",
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context"
]

LABELS = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true"
]


def load_data(split):
    files = list(DATA_DIR.rglob(f"{split}.tsv"))

    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected one {split}.tsv, found {len(files)}"
        )

    df = pd.read_csv(
        files[0],
        sep="\t",
        header=None,
        names=COLUMNS,
        dtype=str
    )

    if df["statement"].isna().any() or df["label"].isna().any():
        raise ValueError(f"Missing data in {split} split")

    if not df["label"].isin(LABELS).all():
        raise ValueError(f"Unexpected labels in {split} split")

    return df


# 1. Load original dataset splits
train = load_data("train")
valid = load_data("valid")

print("Training samples:", len(train))
print("Validation samples:", len(valid))

X_train = train["statement"]
y_train = train["label"]

X_valid = valid["statement"]
y_valid = valid["label"]

# 2. Define TF-IDF + Logistic Regression pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])

# 3. Train on training data only
print("\nTraining Logistic Regression model...")
model.fit(X_train, y_train)

# 4. Evaluate on validation data
predictions = model.predict(X_valid)

accuracy = accuracy_score(y_valid, predictions)
macro_f1 = f1_score(
    y_valid, predictions, average="macro"
)
weighted_f1 = f1_score(
    y_valid, predictions, average="weighted"
)

report = classification_report(
    y_valid,
    predictions,
    labels=LABELS,
    zero_division=0
)

print("\n=== VALIDATION RESULTS ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print("\nClassification report:")
print(report)

# 5. Save metrics
with open(
    RESULT_DIR / "metrics.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("TF-IDF + Logistic Regression\n")
    f.write("Evaluation: Validation split\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
    f.write(report)

# 6. Save confusion matrix as CSV
cm = confusion_matrix(
    y_valid,
    predictions,
    labels=LABELS
)

cm_df = pd.DataFrame(
    cm,
    index=LABELS,
    columns=LABELS
)

cm_df.to_csv(
    RESULT_DIR / "confusion_matrix.csv"
)

# 7. Save validation predictions
results = valid[["id", "statement", "label"]].copy()
results["prediction"] = predictions

results.to_csv(
    RESULT_DIR / "validation_predictions.csv",
    index=False
)

# 8. Save trained model
joblib.dump(
    model,
    MODEL_DIR / "tfidf_logistic_regression.joblib"
)

print("\nTraining and evaluation complete!")
print("Results saved to:", RESULT_DIR)
print("Model saved to:", MODEL_DIR)
