from pathlib import Path
import json
import re
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.sparse import hstack
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
MODEL_DIR = PROJECT_DIR / "models" / "priority"
REPORT_DIR = PROJECT_DIR / "reports" / "priority"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

dataset_files = list(
    RAW_DIR.glob("aa_dataset-tickets-multi-lang*.csv")
)

if not dataset_files:
    raise FileNotFoundError(
        "Priority dataset was not found inside data/raw."
    )

DATASET_PATH = dataset_files[0]

# ---------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------

def clean_text(text):
    text = str(text).lower()

    # Replace sensitive or variable information
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+\.\S+", " email ", text)

    # Keep useful letters, numbers and selected punctuation
    text = re.sub(r"[^a-z0-9!?%$€£\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Keep important negative words
CUSTOM_STOP_WORDS = sorted(
    set(ENGLISH_STOP_WORDS)
    - {
        "no",
        "not",
        "never",
        "without",
        "cannot",
        "couldnt",
        "couldn't",
    }
)

# ---------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------

print("Loading priority dataset...")

df = pd.read_csv(DATASET_PATH, low_memory=False)

required_columns = {"subject", "body", "priority", "language"}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

# English tickets only
df = df[
    df["language"].astype(str).str.lower().eq("en")
].copy()

df["subject"] = df["subject"].fillna("").astype(str)
df["body"] = df["body"].fillna("").astype(str)

df["text"] = (
    df["subject"].str.strip()
    + " "
    + df["body"].str.strip()
).map(clean_text)

df["priority"] = (
    df["priority"]
    .astype(str)
    .str.strip()
    .str.lower()
)

allowed_priorities = {"high", "medium", "low"}

df = df[
    df["priority"].isin(allowed_priorities)
    & df["text"].ne("")
].copy()

# Remove exact duplicate ticket text
before_deduplication = len(df)

df = df.drop_duplicates(
    subset=["text"],
    keep="first",
).reset_index(drop=True)

duplicates_removed = before_deduplication - len(df)

print(f"English usable tickets: {len(df):,}")
print(f"Duplicate tickets removed: {duplicates_removed:,}")

print("\nPriority distribution:")
print(df["priority"].value_counts())

# ---------------------------------------------------------
# Honest train / validation / test split
# ---------------------------------------------------------

train_df, temporary_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["priority"],
)

validation_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=42,
    stratify=temporary_df["priority"],
)

print("\nDATA SPLIT")
print(f"Training tickets: {len(train_df):,}")
print(f"Validation tickets: {len(validation_df):,}")
print(f"Test tickets: {len(test_df):,}")

X_train_text = train_df["text"]
X_validation_text = validation_df["text"]

y_train = train_df["priority"]
y_validation = validation_df["priority"]

# Test data is deliberately not transformed yet.

# ---------------------------------------------------------
# TF-IDF features
# ---------------------------------------------------------

print("\nCreating word TF-IDF features...")

word_vectorizer = TfidfVectorizer(
    stop_words=CUSTOM_STOP_WORDS,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    max_features=100_000,
    sublinear_tf=True,
    strip_accents="unicode",
    dtype=np.float32,
)

X_train_word = word_vectorizer.fit_transform(X_train_text)
X_validation_word = word_vectorizer.transform(
    X_validation_text
)

print("Creating character TF-IDF features...")

character_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=70_000,
    sublinear_tf=True,
    dtype="float32",
)

X_train_character = character_vectorizer.fit_transform(
    X_train_text
)

X_validation_character = character_vectorizer.transform(
    X_validation_text
)

X_train = hstack(
    [X_train_word, X_train_character],
    format="csr",
)

X_validation = hstack(
    [X_validation_word, X_validation_character],
    format="csr",
)

print(f"Total features: {X_train.shape[1]:,}")

# ---------------------------------------------------------
# Candidate models
# ---------------------------------------------------------

candidates = [
    (
        "linear_svm_C0.1",
        LinearSVC(
            C=0.1,
            class_weight=None,
            random_state=42,
        ),
    ),
    (
        "linear_svm_C0.3",
        LinearSVC(
            C=0.3,
            class_weight=None,
            random_state=42,
        ),
    ),
    (
        "linear_svm_C0.5",
        LinearSVC(
            C=0.5,
            class_weight=None,
            random_state=42,
        ),
    ),
    (
        "balanced_svm_C0.1",
        LinearSVC(
            C=0.1,
            class_weight="balanced",
            random_state=42,
        ),
    ),
    (
        "balanced_svm_C0.3",
        LinearSVC(
            C=0.3,
            class_weight="balanced",
            random_state=42,
        ),
    ),
    (
        "logistic_regression",
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        ),
    ),
]

results = []
best_model = None
best_model_name = None
best_macro_f1 = -1
best_accuracy = -1

print("\nTraining candidate models...")

for number, (model_name, model) in enumerate(
    candidates,
    start=1,
):
    print(
        f"\n[{number}/{len(candidates)}] "
        f"Training {model_name}..."
    )

    start_time = time.time()

    model.fit(X_train, y_train)
    validation_predictions = model.predict(X_validation)

    accuracy = accuracy_score(
        y_validation,
        validation_predictions,
    )

    macro_f1 = f1_score(
        y_validation,
        validation_predictions,
        average="macro",
    )

    elapsed = time.time() - start_time

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1 * 100:.2f}%")
    print(f"Time: {elapsed:.1f} seconds")

    results.append({
        "model": model_name,
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1,
        "training_seconds": elapsed,
    })

    if (
        macro_f1 > best_macro_f1
        or (
            macro_f1 == best_macro_f1
            and accuracy > best_accuracy
        )
    ):
        best_model = model
        best_model_name = model_name
        best_macro_f1 = macro_f1
        best_accuracy = accuracy

# Save validation comparison
results_df = pd.DataFrame(results).sort_values(
    by=["validation_macro_f1", "validation_accuracy"],
    ascending=False,
)

results_df.to_csv(
    REPORT_DIR / "validation_comparison.csv",
    index=False,
)

print("\n" + "=" * 60)
print("VALIDATION COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

print("\nSelected model:", best_model_name)

# ---------------------------------------------------------
# Final test evaluation — loaded only after selection
# ---------------------------------------------------------

print("\nEvaluating the selected model on untouched test data...")

X_test_word = word_vectorizer.transform(test_df["text"])
X_test_character = character_vectorizer.transform(
    test_df["text"]
)

X_test = hstack(
    [X_test_word, X_test_character],
    format="csr",
)

y_test = test_df["priority"]
test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_predictions)

test_precision = precision_score(
    y_test,
    test_predictions,
    average="macro",
)

test_recall = recall_score(
    y_test,
    test_predictions,
    average="macro",
)

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
)

test_weighted_f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted",
)

majority_accuracy = (
    y_test.value_counts(normalize=True).max()
)

print("\n" + "=" * 60)
print("FINAL PRIORITY TEST RESULTS")
print("=" * 60)
print(f"Selected model: {best_model_name}")
print(f"Test tickets: {len(test_df):,}")
print(f"Accuracy: {test_accuracy * 100:.2f}%")
print(f"Macro precision: {test_precision * 100:.2f}%")
print(f"Macro recall: {test_recall * 100:.2f}%")
print(f"Macro F1: {test_macro_f1 * 100:.2f}%")
print(f"Weighted F1: {test_weighted_f1 * 100:.2f}%")
print(
    f"Majority baseline: "
    f"{majority_accuracy * 100:.2f}%"
)

labels = ["high", "medium", "low"]

report_text = classification_report(
    y_test,
    test_predictions,
    labels=labels,
    digits=3,
)

print("\nPER-PRIORITY RESULTS")
print(report_text)

with open(
    REPORT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8",
) as report_file:
    report_file.write(report_text)

# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=labels,
)

matrix_df = pd.DataFrame(
    matrix,
    index=labels,
    columns=labels,
)

matrix_df.to_csv(
    REPORT_DIR / "confusion_matrix.csv"
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    matrix_df,
    annot=True,
    fmt="d",
    cmap="Blues",
)

plt.title("Priority Model Confusion Matrix")
plt.xlabel("Predicted priority")
plt.ylabel("Actual priority")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "confusion_matrix.png",
    dpi=200,
)

plt.close()

# ---------------------------------------------------------
# Save complete model system
# ---------------------------------------------------------

model_package = {
    "word_vectorizer": word_vectorizer,
    "character_vectorizer": character_vectorizer,
    "classifier": best_model,
    "model_name": best_model_name,
    "labels": labels,
}

model_path = MODEL_DIR / "priority_model.joblib"

joblib.dump(model_package, model_path)

model_information = {
    "dataset": DATASET_PATH.name,
    "language": "English",
    "input_fields": ["subject", "body"],
    "training_tickets": len(train_df),
    "validation_tickets": len(validation_df),
    "test_tickets": len(test_df),
    "model": best_model_name,
    "test_accuracy": round(test_accuracy, 6),
    "test_macro_precision": round(test_precision, 6),
    "test_macro_recall": round(test_recall, 6),
    "test_macro_f1": round(test_macro_f1, 6),
    "test_weighted_f1": round(test_weighted_f1, 6),
    "majority_baseline_accuracy": round(
        majority_accuracy,
        6,
    ),
}

with open(
    MODEL_DIR / "priority_model_info.json",
    "w",
    encoding="utf-8",
) as information_file:
    json.dump(
        model_information,
        information_file,
        indent=4,
    )

print("\nMODEL SAVED")
print(model_path)

print("\nREPORTS SAVED")
print(REPORT_DIR)

print(
    "\nImportant: Report the final test accuracy honestly. "
    "Do not use validation accuracy as the final result."
)