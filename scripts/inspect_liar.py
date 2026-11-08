
from pathlib import Path
import pandas as pd

root = Path("data/raw/liar")
columns = [
    "id", "label", "statement", "subject", "speaker",
    "job_title", "state_info", "party_affiliation",
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context"
]

for name in ["train", "valid", "test"]:
    matches = list(root.rglob(f"{name}.tsv"))
    if not matches:
        raise FileNotFoundError(f"Missing {name}.tsv")

    df = pd.read_csv(
        matches[0],
        sep="\t",
        header=None,
        names=columns
    )

    print(f"\n{name.upper()}: {len(df)} rows")
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print("\nMissing statements:", df["statement"].isna().sum())
    print("\nExample:")
    print(df[["statement", "label"]].head(3).to_string(index=False))
