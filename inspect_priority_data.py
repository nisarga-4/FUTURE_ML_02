from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"

files = list(RAW_DIR.glob("aa_dataset-tickets-multi-lang*.csv"))

if not files:
    raise FileNotFoundError(
        "Priority dataset not found inside data/raw."
    )

file_path = files[0]

print(f"Reading: {file_path.name}")
print("Please wait...\n")

df = pd.read_csv(file_path, low_memory=False)

print("=" * 60)
print("BASIC INFORMATION")
print("=" * 60)
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nCOLUMN NAMES")
for column in df.columns:
    print(f"- {column}")

print("\nMISSING VALUES")
print(df.isna().sum().sort_values(ascending=False).head(20))

# Find important columns safely
column_lookup = {
    str(column).strip().lower(): column
    for column in df.columns
}

def find_column(possible_names):
    for name in possible_names:
        if name.lower() in column_lookup:
            return column_lookup[name.lower()]
    return None

subject_column = find_column(
    ["subject", "ticket subject", "title"]
)

body_column = find_column(
    ["body", "ticket body", "description", "text"]
)

priority_column = find_column(
    ["priority", "ticket priority", "urgency"]
)

language_column = find_column(
    ["language", "lang"]
)

print("\nDETECTED IMPORTANT COLUMNS")
print(f"Subject column: {subject_column}")
print(f"Body column: {body_column}")
print(f"Priority column: {priority_column}")
print(f"Language column: {language_column}")

if priority_column:
    print("\nPRIORITY DISTRIBUTION")
    print(
        df[priority_column]
        .fillna("MISSING")
        .astype(str)
        .str.strip()
        .value_counts(dropna=False)
    )

if language_column:
    print("\nLANGUAGE DISTRIBUTION")
    print(
        df[language_column]
        .fillna("MISSING")
        .astype(str)
        .str.strip()
        .value_counts(dropna=False)
        .head(20)
    )

# Combine subject and body for duplicate inspection
if subject_column or body_column:
    subject = (
        df[subject_column].fillna("").astype(str)
        if subject_column
        else pd.Series("", index=df.index)
    )

    body = (
        df[body_column].fillna("").astype(str)
        if body_column
        else pd.Series("", index=df.index)
    )

    combined_text = (
        subject.str.strip() + " " + body.str.strip()
    ).str.strip()

    normalized_text = (
        combined_text
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    print("\nTEXT QUALITY")
    print(f"Empty ticket texts: {(normalized_text == '').sum():,}")
    print(
        "Repeated texts beyond first occurrence: "
        f"{normalized_text.duplicated().sum():,}"
    )

    if priority_column:
        audit = pd.DataFrame({
            "text": normalized_text,
            "priority": (
                df[priority_column]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )
        })

        audit = audit[
            (audit["text"] != "") &
            (audit["priority"] != "")
        ]

        conflicting_groups = (
            audit.groupby("text")["priority"]
            .nunique()
            .gt(1)
            .sum()
        )

        print(
            "Text groups with conflicting priorities: "
            f"{conflicting_groups:,}"
        )

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("The original dataset was not modified.")
print("=" * 60)