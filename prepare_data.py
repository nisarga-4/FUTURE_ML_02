from pathlib import Path
import pandas as pd

# Find folders relative to this script.
PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"


def normalize_text(series):
    """Standardize letter case and whitespace."""
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_dataset(filename, required_columns):
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find:\n{path}\n"
            "Check that the CSV is inside data/raw."
        )

    df = pd.read_csv(path)

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{filename} is missing these columns: {missing}"
        )

    return df


def remove_invalid_and_repeated_rows(df):
    valid = (
        df["ticket_text"].ne("")
        & df["category"].notna()
        & df["category"].ne("")
    )

    return (
        df.loc[valid]
        .drop_duplicates(subset=["ticket_text"])
        .reset_index(drop=True)
    )


def main():
    # 1. Load original datasets.
    it_raw = load_dataset(
        "all_tickets_processed_improved_v3.csv",
        ["Document", "Topic_group"],
    )

    customer_raw = load_dataset(
        "customer_support_tickets.csv",
        [
            "Ticket Subject",
            "Ticket Description",
            "Ticket Type",
            "Ticket Priority",
        ],
    )

    # 2. Give both datasets consistent column names.
    it = pd.DataFrame({
        "ticket_text": normalize_text(it_raw["Document"]),
        "category": it_raw["Topic_group"].str.strip(),
        # This dataset has no priority labels.
        "priority": pd.NA,
        "source_dataset": "internal_it",
    })

    customer = pd.DataFrame({
        "ticket_text": normalize_text(
            customer_raw["Ticket Subject"].fillna("")
            + " "
            + customer_raw["Ticket Description"].fillna("")
        ),
        "category": customer_raw["Ticket Type"].str.strip(),
        "priority": customer_raw["Ticket Priority"].str.strip(),
        "source_dataset": "customer_support",
    })

    # 3. Find identical customer texts with conflicting labels.
    # Check original priorities before combining Critical and High.
    category_counts = (
        customer.groupby("ticket_text")["category"].nunique()
    )
    priority_counts = (
        customer.groupby("ticket_text")["priority"].nunique()
    )

    conflicting_texts = (
        set(category_counts[category_counts > 1].index)
        | set(priority_counts[priority_counts > 1].index)
    )

    needs_review = customer["ticket_text"].isin(conflicting_texts)

    review = customer.loc[needs_review].copy()
    customer = customer.loc[~needs_review].copy()

    # 4. Remove empty inputs, missing categories, and exact repeats.
    it = remove_invalid_and_repeated_rows(it)
    customer = remove_invalid_and_repeated_rows(customer)

    # Match the internship's three priority levels.
    customer["priority"] = customer["priority"].replace({
        "Critical": "High",
    })

    # 5. Preserve category meanings across the two sources.
    it["category_id"] = "internal_it::" + it["category"]
    customer["category_id"] = (
        "customer_support::" + customer["category"]
    )

    combined = pd.concat([it, customer], ignore_index=True)

    # 6. Save new files. Original datasets remain unchanged.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    it.to_csv(
        OUTPUT_DIR / "internal_it_prepared.csv",
        index=False,
    )
    customer.to_csv(
        OUTPUT_DIR / "customer_support_prepared.csv",
        index=False,
    )
    combined.to_csv(
        OUTPUT_DIR / "combined_candidate.csv",
        index=False,
    )
    review.to_csv(
        OUTPUT_DIR / "customer_conflicts_for_review.csv",
        index=False,
    )

    print("\nPREPARATION COMPLETE")
    print(f"Internal IT tickets retained: {len(it):,}")
    print(f"Customer tickets retained: {len(customer):,}")
    print(f"Customer rows set aside for review: {len(review):,}")
    print(f"Combined candidate rows: {len(combined):,}")
    print(f"Category labels: {combined['category_id'].nunique()}")
    print(
        "Rows without priority: "
        f"{combined['priority'].isna().sum():,}"
    )
    print(f"\nSaved files in: {OUTPUT_DIR}")
    print("Original CSVs were not modified.")


if __name__ == "__main__":
    main()