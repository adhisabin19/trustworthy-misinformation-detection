
from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = Path("data/raw/liar")
RESULT_DIR = Path("reports/baseline")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "job_title", "state_info", "party_affiliation",
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context"
]

def load_split(name):
    paths = list(DATA_DIR.rglob(f"{name}.tsv"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one {name}.tsv")

    return pd.read_csv(
        paths[0],
        sep="\t",
        header=None,
        names=COLUMNS,
        dtype=str
    )

train = load_split("train")
valid = load_split("valid")

X_train = train["statement"].fillna("")
y_train = train["label"]
X_valid = valid["statement"].fillna("")
y_valid = valid["label"]

models = {
    "Dummy Majority": DummyClassifier(
        strategy="most_frequent"
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=5000
    )
}

results = []

for name, classifier in models.items():
    print(f"\nTraining: {name}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
            sublinear_tf=True
        )),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_valid)

    accuracy = accuracy_score(y_valid, predictions)
    macro_f1 = f1_score(
        y_valid,
        predictions,
        average="macro",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Macro_F1": round(macro_f1, 4)
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "Macro_F1",
    ascending=False
)

print("\n=== BASELINE COMPARISON ===")
print(results_df.to_string(index=False))

results_df.to_csv(
    RESULT_DIR / "model_comparison.csv",
    index=False
)

print("\nComparison saved successfully.")
