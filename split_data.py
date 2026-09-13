from pathlib import Path
import hashlib
import re

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "combined_candidate.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "splits"


def make_group(text):
    """Group texts that differ only in numbers, punctuation, or spacing."""
    text = str(text).lower()
    text = re.sub(r"\d+", " numbertoken ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}\nRun prepare_data.py first."
        )

    # Prevent accidental replacement of an established evaluation split.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR} already contains files. "
            "Keep the existing splits for consistent evaluation."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {"ticket_text", "category_id", "source_dataset"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing text, category, or source values found.")

    if df["ticket_text"].str.strip().eq("").any():
        raise ValueError("Empty ticket text found.")

    df["group_id"] = df["ticket_text"].map(make_group)

    # Require enough independent groups for each category.
    groups_per_category = df.groupby("category_id")["group_id"].nunique()
    if groups_per_category.min() < 10:
        raise ValueError(
            "At least one category has fewer than 10 text groups. "
            "Share this output before continuing:\n"
            + groups_per_category.to_string()
        )

    # Create 10 folds, preserving text groups and approximately
    # preserving category proportions.
    splitter = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=42,
    )

    df["fold"] = -1

    for fold_number, (_, held_out_indices) in enumerate(
        splitter.split(
            X=df["ticket_text"],
            y=df["category_id"],
            groups=df["group_id"],
        )
    ):
        df.loc[held_out_indices, "fold"] = fold_number

    # Fixed assignments: never choose folds based on model scores.
    df["split"] = "train"
    df.loc[df["fold"].eq(2), "split"] = "validation"
    df.loc[df["fold"].isin([0, 1]), "split"] = "test"

    # Verify every text group appears in exactly one split.
    if df.groupby("group_id")["split"].nunique().max() != 1:
        raise RuntimeError("A text group crosses split boundaries.")

    # Check all categories are present in each split.
    all_categories = set(df["category_id"])
    for split_name in ["train", "validation", "test"]:
        part = df.loc[df["split"].eq(split_name)]
        absent = all_categories - set(part["category_id"])
        if absent:
            raise ValueError(
                f"{split_name} is missing categories: {sorted(absent)}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Derive all experiment files from the SAME split assignments.
    for split_name in ["train", "validation", "test"]:
        combined = df.loc[df["split"].eq(split_name)].copy()

        internal_it = combined.loc[
            combined["source_dataset"].eq("internal_it")
        ].copy()

        customer = combined.loc[
            combined["source_dataset"].eq("customer_support")
        ].copy()

        combined.to_csv(
            OUTPUT_DIR / f"combined_{split_name}.csv",
            index=False,
        )
        internal_it.to_csv(
            OUTPUT_DIR / f"internal_it_{split_name}.csv",
            index=False,
        )
        customer.to_csv(
            OUTPUT_DIR / f"customer_support_{split_name}.csv",
            index=False,
        )

    summary = pd.crosstab(df["split"], df["source_dataset"]).reindex(
        ["train", "validation", "test"]
    )
    summary["total"] = summary.sum(axis=1)

    category_counts = pd.crosstab(
        df["category_id"], df["split"]
    ).reindex(columns=["train", "validation", "test"])

    summary.to_csv(OUTPUT_DIR / "split_summary.csv")
    category_counts.to_csv(OUTPUT_DIR / "category_counts.csv")

    print("\nSPLITTING COMPLETE")
    print(summary.to_string())

    print(f"\nUnique text groups: {df['group_id'].nunique():,}")
    print("Text-group overlap between splits: 0")
    print("All categories are present in every split.")
    print(f"\nSaved files in: {OUTPUT_DIR}")
    print("No model has been trained yet.")


if __name__ == "__main__":
    main()