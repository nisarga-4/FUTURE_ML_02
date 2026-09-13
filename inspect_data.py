from pathlib import Path
import pandas as pd

# Locate the raw-data folder relative to this Python file.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw"

datasets = [
    (
        "all_tickets_processed_improved_v3.csv",
        ["Document"],
        ["Topic_group"],
    ),
    (
        "customer_support_tickets.csv",
        ["Ticket Subject", "Ticket Description"],
        ["Ticket Type", "Ticket Priority"],
    ),
]

for filename, text_columns, label_columns in datasets:
    print("\n" + "=" * 60)
    print(f"FILE: {filename}")

    file_path = DATA_DIR / filename

    if not file_path.exists():
        print(f"File not found: {file_path}")
        print("Check the filename and its location.")
        continue

    df = pd.read_csv(file_path)

    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumn names:")
    print(df.columns.tolist())

    required = text_columns + label_columns
    missing_columns = [col for col in required if col not in df.columns]

    if missing_columns:
        print(f"Expected columns missing: {missing_columns}")
        continue

    print("\nMissing values in text and label columns:")
    print(df[required].isna().sum().to_string())

    # Combine text columns and normalize case and whitespace.
    text = (
        df[text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    nonempty = text.ne("")
    print(f"\nEmpty combined texts: {(~nonempty).sum():,}")
    print(
        "Repeated nonempty texts beyond their first occurrence: "
        f"{text[nonempty].duplicated().sum():,}"
    )

    for label in label_columns:
        print(f"\nLabel counts — {label}:")
        print(df[label].value_counts(dropna=False).to_string())

        # Check whether identical input text has different labels.
        pairs = pd.DataFrame({
            "text": text,
            "label": df[label],
        })

        pairs = pairs.loc[nonempty].dropna(subset=["label"])
        label_counts = pairs.groupby("text")["label"].nunique()
        conflicts = (label_counts > 1).sum()

        print(f"Text groups with conflicting labels: {conflicts:,}")

print("\nInspection complete. Original files were not changed.")