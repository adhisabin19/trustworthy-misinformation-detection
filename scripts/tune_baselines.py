
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = Path("data/raw/liar")
OUTPUT_DIR = Path("reports/tuning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

experiments = []

feature_settings = [
    {"ngram_range": (1, 1), "max_features": 10000},
    {"ngram_range": (1, 2), "max_features": 20000},
    {"ngram_range": (1, 3), "max_features": 30000},
]

for features in feature_settings:
    for model_name in ["Logistic Regression", "Linear SVM"]:
        for C in [0.1, 1.0, 10.0]:

            if model_name == "Logistic Regression":
                classifier = LogisticRegression(
                    C=C,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42
                )
            else:
                classifier = LinearSVC(
                    C=C,
                    class_weight="balanced",
                    random_state=42,
                    max_iter=5000
                )

            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=features["ngram_range"],
                    max_features=features["max_features"],
                    min_df=2,
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

            experiments.append({
                "model": model_name,
                "ngram": str(features["ngram_range"]),
                "max_features": features["max_features"],
                "C": C,
                "accuracy": accuracy,
                "macro_f1": macro_f1
            })

            print(
                f"{model_name} | "
                f"ngram={features['ngram_range']} | "
                f"C={C} | "
                f"Macro-F1={macro_f1:.4f}"
            )

results = pd.DataFrame(experiments)
results = results.sort_values(
    "macro_f1",
    ascending=False
)

results.to_csv(
    OUTPUT_DIR / "tuning_results.csv",
    index=False
)

print("\n=== TOP 5 CONFIGURATIONS ===")
print(results.head(5).to_string(index=False))

print("\nTuning complete!")
