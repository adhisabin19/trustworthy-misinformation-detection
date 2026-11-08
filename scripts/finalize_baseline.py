
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_DIR = Path("data/raw/liar")
OUTPUT_DIR = Path("reports/tuned_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "job_title", "state_info", "party_affiliation",
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context"
]

def load_split(name):
    matches = list(DATA_DIR.rglob(f"{name}.tsv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name}.tsv")
    return pd.read_csv(
        matches[0],
        sep="\t",
        header=None,
        names=COLUMNS,
        dtype=str
    )

train = load_split("train")
valid = load_split("valid")

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        C=10.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

print("Training selected Logistic Regression...")
model.fit(train["statement"], train["label"])

predictions = model.predict(valid["statement"])

metrics = {
    "model": "TF-IDF + Logistic Regression",
    "dataset": "LIAR",
    "evaluation_split": "validation",
    "accuracy": float(accuracy_score(valid["label"], predictions)),
    "macro_f1": float(f1_score(
        valid["label"], predictions, average="macro"
    )),
    "weighted_f1": float(f1_score(
        valid["label"], predictions, average="weighted"
    )),
    "parameters": {
        "ngram_range": [1, 2],
        "max_features": 20000,
        "min_df": 2,
        "sublinear_tf": True,
        "C": 10.0,
        "class_weight": "balanced",
        "random_state": 42
    }
}

with open(
    OUTPUT_DIR / "metrics.json", "w", encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=2)

joblib.dump(
    model,
    OUTPUT_DIR / "tuned_logistic_regression.joblib"
)

results = valid[["id", "statement", "label"]].copy()
results["prediction"] = predictions
results["correct"] = results["label"] == results["prediction"]

results.to_csv(
    OUTPUT_DIR / "validation_predictions.csv",
    index=False
)

print("\n=== SELECTED MODEL ===")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro-F1: {metrics['macro_f1']:.4f}")
print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
print("\n", classification_report(
    valid["label"], predictions, zero_division=0
))
print("\nTuned baseline saved successfully!")
