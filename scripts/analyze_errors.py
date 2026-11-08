
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DATA_DIR = Path("data/raw/liar")
OUTPUT_DIR = Path("reports/error_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "job_title", "state_info", "party_affiliation",
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context"
]

LABELS = [
    "pants-fire", "false", "barely-true",
    "half-true", "mostly-true", "true"
]

def load_split(name):
    paths = list(DATA_DIR.rglob(f"{name}.tsv"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one {name}.tsv")

    return pd.read_csv(
        paths[0], sep="\t", header=None,
        names=COLUMNS, dtype=str
    )

train = load_split("train")
valid = load_split("valid")

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        sublinear_tf=True
    )),
    ("classifier", LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=5000
    ))
])

model.fit(train["statement"], train["label"])

predictions = model.predict(valid["statement"])

# Export individual predictions
results = valid[["id", "statement", "label"]].copy()
results["prediction"] = predictions
results["correct"] = results["label"] == results["prediction"]

results.to_csv(
    OUTPUT_DIR / "all_predictions.csv",
    index=False
)

errors = results[~results["correct"]]
errors.to_csv(
    OUTPUT_DIR / "misclassified_statements.csv",
    index=False
)

print("Total statements:", len(results))
print("Correct predictions:", results["correct"].sum())
print("Incorrect predictions:", len(errors))

# Confusion matrix
cm = confusion_matrix(
    valid["label"], predictions, labels=LABELS
)

fig, ax = plt.subplots(figsize=(10, 8))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=LABELS
).plot(ax=ax, cmap="Blues", xticks_rotation=45)

ax.set_title("TrustFact — Linear SVM Confusion Matrix")
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=200
)

plt.close(fig)

# Most frequent misclassifications
pairs = (
    errors.groupby(["label", "prediction"])
    .size()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 misclassification patterns:")
print(pairs)

print("\nError analysis complete!")
