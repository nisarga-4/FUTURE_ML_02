from pathlib import Path
import json
import re
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# Paths
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
MODEL_DIR = PROJECT_DIR / "models" / "priority_v2"
REPORT_DIR = PROJECT_DIR / "reports" / "priority_v2"

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

# =========================================================
# Text cleaning
# =========================================================

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+\.\S+", " email ", text)
    text = re.sub(r"\b\d{7,}\b", " number ", text)

    # Preserve urgency-related punctuation
    text = re.sub(r"[^a-z0-9!?%$€£\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


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
        "nothing",
        "nobody",
    }
)

# =========================================================
# Engineered severity features
# =========================================================

HIGH_URGENCY_WORDS = [
    "urgent",
    "urgently",
    "immediate",
    "immediately",
    "critical",
    "emergency",
    "severe",
    "priority",
    "asap",
]

OUTAGE_WORDS = [
    "outage",
    "offline",
    "system down",
    "service down",
    "unavailable",
    "disruption",
    "failure",
    "crash",
    "not working",
]

SECURITY_WORDS = [
    "security breach",
    "breach",
    "hacked",
    "malware",
    "ransomware",
    "unauthorized",
    "phishing",
    "compromised",
    "data leak",
]

DATA_LOSS_WORDS = [
    "data loss",
    "lost data",
    "deleted data",
    "corrupted",
    "cannot recover",
    "backup failed",
]

BLOCKING_WORDS = [
    "blocked",
    "cannot access",
    "unable to access",
    "cannot login",
    "unable to login",
    "cannot work",
    "stopped working",
]

MULTI_USER_WORDS = [
    "all users",
    "entire team",
    "whole team",
    "multiple users",
    "several users",
    "company wide",
    "organization wide",
    "everyone",
]

LOW_URGENCY_WORDS = [
    "general inquiry",
    "information",
    "documentation",
    "feature request",
    "suggestion",
    "when possible",
    "minor issue",
    "question about",
]


def keyword_count(text, keywords):
    return sum(text.count(keyword) for keyword in keywords)


def create_severity_features(frame):
    rows = []

    for subject, body in zip(
        frame["subject"],
        frame["body"],
    ):
        original_text = f"{subject} {body}"
        lower_text = original_text.lower()

        features = [
            min(
                keyword_count(lower_text, HIGH_URGENCY_WORDS),
                5,
            ) / 5,
            min(
                keyword_count(lower_text, OUTAGE_WORDS),
                5,
            ) / 5,
            min(
                keyword_count(lower_text, SECURITY_WORDS),
                5,
            ) / 5,
            min(
                keyword_count(lower_text, DATA_LOSS_WORDS),
                5,
            ) / 5,
            min(
                keyword_count(lower_text, BLOCKING_WORDS),
                5,
            ) / 5,
            min(
                keyword_count(lower_text, MULTI_USER_WORDS),
                5,
            ) / 5,
            min(
                keyword_count(lower_text, LOW_URGENCY_WORDS),
                5,
            ) / 5,
            min(original_text.count("!"), 5) / 5,
            min(original_text.count("?"), 5) / 5,
            min(len(original_text.split()), 500) / 500,
        ]

        rows.append(features)

    # Multiply so these business signals receive useful weight
    return csr_matrix(
        np.asarray(rows, dtype=np.float32) * 2.0
    )


# =========================================================
# Load data
# =========================================================

print("Loading priority dataset...")

df = pd.read_csv(DATASET_PATH, low_memory=False)

required_columns = {
    "subject",
    "body",
    "priority",
    "language",
    "type",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

df = df[
    df["language"]
    .fillna("")
    .astype(str)
    .str.lower()
    .eq("en")
].copy()

for column in ["subject", "body", "type"]:
    df[column] = (
        df[column]
        .fillna("unknown")
        .astype(str)
        .str.strip()
    )

df["priority"] = (
    df["priority"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

df = df[
    df["priority"].isin({"high", "medium", "low"})
].copy()

# Give the subject three times its normal importance
df["model_text"] = (
    df["subject"]
    + " "
    + df["subject"]
    + " "
    + df["subject"]
    + " "
    + df["body"]
).map(clean_text)

df["duplicate_key"] = (
    df["subject"].str.lower().str.strip()
    + " "
    + df["body"].str.lower().str.strip()
).str.replace(r"\s+", " ", regex=True)

df = df[
    df["model_text"].ne("")
].drop_duplicates(
    subset=["duplicate_key"],
    keep="first",
).reset_index(drop=True)

print(f"Usable English tickets: {len(df):,}")

print("\nPriority distribution:")
print(df["priority"].value_counts())

print("\nTicket-type distribution:")
print(df["type"].value_counts())

# =========================================================
# Train, validation and test split
# =========================================================

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
print(f"Training: {len(train_df):,}")
print(f"Validation: {len(validation_df):,}")
print(f"Test: {len(test_df):,}")

y_train = train_df["priority"]
y_validation = validation_df["priority"]

# =========================================================
# Word TF-IDF
# =========================================================

print("\nCreating word and phrase features...")

word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    stop_words=CUSTOM_STOP_WORDS,
    min_df=2,
    max_df=0.99,
    max_features=130_000,
    sublinear_tf=True,
    strip_accents="unicode",
    dtype=np.float32,
)

X_train_word = word_vectorizer.fit_transform(
    train_df["model_text"]
)

X_validation_word = word_vectorizer.transform(
    validation_df["model_text"]
)

# =========================================================
# Character TF-IDF
# =========================================================

print("Creating character features...")

character_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=90_000,
    sublinear_tf=True,
    dtype=np.float32,
)

X_train_character = character_vectorizer.fit_transform(
    train_df["model_text"]
)

X_validation_character = character_vectorizer.transform(
    validation_df["model_text"]
)

# =========================================================
# Ticket-type feature
# =========================================================

print("Creating ticket-type features...")

type_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True,
    dtype=np.float32,
)

X_train_type = type_encoder.fit_transform(
    train_df[["type"]]
)

X_validation_type = type_encoder.transform(
    validation_df[["type"]]
)

# =========================================================
# Severity features
# =========================================================

print("Creating severity and urgency features...")

X_train_severity = create_severity_features(train_df)
X_validation_severity = create_severity_features(
    validation_df
)

X_train = hstack(
    [
        X_train_word,
        X_train_character,
        X_train_type,
        X_train_severity,
    ],
    format="csr",
)

X_validation = hstack(
    [
        X_validation_word,
        X_validation_character,
        X_validation_type,
        X_validation_severity,
    ],
    format="csr",
)

print(f"Total features: {X_train.shape[1]:,}")

# =========================================================
# Candidate models
# =========================================================

candidates = [
    (
        "svm_C0.25",
        LinearSVC(C=0.25, random_state=42),
    ),
    (
        "svm_C0.50",
        LinearSVC(C=0.50, random_state=42),
    ),
    (
        "svm_C0.75",
        LinearSVC(C=0.75, random_state=42),
    ),
    (
        "svm_C1.00",
        LinearSVC(C=1.00, random_state=42),
    ),
    (
        "svm_C1.50",
        LinearSVC(C=1.50, random_state=42),
    ),
    (
        "balanced_svm_C0.50",
        LinearSVC(
            C=0.50,
            class_weight="balanced",
            random_state=42,
        ),
    ),
    (
        "balanced_svm_C0.75",
        LinearSVC(
            C=0.75,
            class_weight="balanced",
            random_state=42,
        ),
    ),
    (
        "logistic_regression",
        LogisticRegression(
            C=2.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        ),
    ),
    (
        "sgd_modified_huber",
        SGDClassifier(
            loss="modified_huber",
            alpha=0.00001,
            max_iter=2000,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        ),
    ),
    (
        "complement_naive_bayes",
        ComplementNB(alpha=0.1),
    ),
]

results = []
trained_models = {}

best_model = None
best_name = None
best_macro_f1 = -1
best_accuracy = -1

print("\nTraining advanced priority models...")

for number, (name, model) in enumerate(
    candidates,
    start=1,
):
    print(
        f"\n[{number}/{len(candidates)}] Training {name}..."
    )

    started = time.time()

    try:
        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)

        accuracy = accuracy_score(
            y_validation,
            predictions,
        )

        macro_f1 = f1_score(
            y_validation,
            predictions,
            average="macro",
        )

        elapsed = time.time() - started

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Macro F1: {macro_f1 * 100:.2f}%")
        print(f"Time: {elapsed:.1f} seconds")

        trained_models[name] = model

        results.append({
            "model": name,
            "validation_accuracy": accuracy,
            "validation_macro_f1": macro_f1,
            "training_seconds": elapsed,
            "status": "completed",
        })

        if (
            macro_f1 > best_macro_f1
            or (
                np.isclose(macro_f1, best_macro_f1)
                and accuracy > best_accuracy
            )
        ):
            best_model = model
            best_name = name
            best_macro_f1 = macro_f1
            best_accuracy = accuracy

    except Exception as error:
        print(f"Model failed: {error}")

        results.append({
            "model": name,
            "validation_accuracy": np.nan,
            "validation_macro_f1": np.nan,
            "training_seconds": time.time() - started,
            "status": f"failed: {error}",
        })

if best_model is None:
    raise RuntimeError("Every candidate model failed.")

results_df = pd.DataFrame(results).sort_values(
    by=["validation_macro_f1", "validation_accuracy"],
    ascending=False,
    na_position="last",
)

results_df.to_csv(
    REPORT_DIR / "validation_comparison.csv",
    index=False,
)

print("\n" + "=" * 65)
print("VALIDATION COMPARISON")
print("=" * 65)
print(results_df.to_string(index=False))

print(f"\nSelected model: {best_name}")
print(
    f"Validation accuracy: {best_accuracy * 100:.2f}%"
)
print(
    f"Validation macro F1: {best_macro_f1 * 100:.2f}%"
)

# =========================================================
# Confidence margin
# =========================================================

def calculate_margins(model, features):
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)

    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)

    else:
        raise ValueError(
            "Selected model has no confidence scores."
        )

    scores = np.asarray(scores)

    sorted_scores = np.sort(scores, axis=1)

    return sorted_scores[:, -1] - sorted_scores[:, -2]


validation_predictions = best_model.predict(
    X_validation
)

validation_margins = calculate_margins(
    best_model,
    X_validation,
)

# Choose maximum coverage that achieves at least 85%
target_accuracy = 0.85

threshold_candidates = np.unique(
    np.quantile(
        validation_margins,
        np.linspace(0, 0.99, 250),
    )
)

threshold_results = []

minimum_automatic_tickets = max(
    50,
    int(len(validation_df) * 0.10),
)

for threshold in threshold_candidates:
    automatic_mask = validation_margins >= threshold
    automatic_count = automatic_mask.sum()

    if automatic_count < minimum_automatic_tickets:
        continue

    automatic_accuracy = accuracy_score(
        y_validation[automatic_mask],
        validation_predictions[automatic_mask],
    )

    coverage = automatic_count / len(validation_df)

    threshold_results.append({
        "threshold": float(threshold),
        "automatic_accuracy": automatic_accuracy,
        "coverage": coverage,
        "automatic_tickets": int(automatic_count),
    })

threshold_df = pd.DataFrame(threshold_results)

acceptable_thresholds = threshold_df[
    threshold_df["automatic_accuracy"] >= target_accuracy
].sort_values(
    by=["coverage", "automatic_accuracy"],
    ascending=False,
)

if len(acceptable_thresholds) > 0:
    selected_threshold_row = acceptable_thresholds.iloc[0]
else:
    selected_threshold_row = threshold_df.sort_values(
        by=["automatic_accuracy", "coverage"],
        ascending=False,
    ).iloc[0]

selected_threshold = float(
    selected_threshold_row["threshold"]
)

threshold_df.to_csv(
    REPORT_DIR / "confidence_thresholds.csv",
    index=False,
)

print("\nSELECTIVE PRIORITY CALIBRATION")
print(f"Decision threshold: {selected_threshold:.4f}")
print(
    "Validation automatic accuracy: "
    f"{selected_threshold_row['automatic_accuracy'] * 100:.2f}%"
)
print(
    "Validation automation coverage: "
    f"{selected_threshold_row['coverage'] * 100:.2f}%"
)

# =========================================================
# Untouched test evaluation
# =========================================================

print("\nTransforming untouched test tickets...")

X_test_word = word_vectorizer.transform(
    test_df["model_text"]
)

X_test_character = character_vectorizer.transform(
    test_df["model_text"]
)

X_test_type = type_encoder.transform(
    test_df[["type"]]
)

X_test_severity = create_severity_features(test_df)

X_test = hstack(
    [
        X_test_word,
        X_test_character,
        X_test_type,
        X_test_severity,
    ],
    format="csr",
)

y_test = test_df["priority"]

test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_predictions,
)

test_precision = precision_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0,
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

majority_baseline = (
    y_test.value_counts(normalize=True).max()
)

print("\n" + "=" * 65)
print("FINAL PRIORITY V2 TEST RESULTS")
print("=" * 65)
print(f"Selected model: {best_name}")
print(f"Test tickets: {len(test_df):,}")
print(f"Accuracy: {test_accuracy * 100:.2f}%")
print(
    f"Macro precision: {test_precision * 100:.2f}%"
)
print(f"Macro recall: {test_recall * 100:.2f}%")
print(f"Macro F1: {test_macro_f1 * 100:.2f}%")
print(f"Weighted F1: {test_weighted_f1 * 100:.2f}%")
print(
    f"Majority baseline: {majority_baseline * 100:.2f}%"
)

labels = ["high", "medium", "low"]

report = classification_report(
    y_test,
    test_predictions,
    labels=labels,
    digits=3,
    zero_division=0,
)

print("\nPER-PRIORITY RESULTS")
print(report)

with open(
    REPORT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8",
) as report_file:
    report_file.write(report)

# =========================================================
# Selective test performance
# =========================================================

test_margins = calculate_margins(
    best_model,
    X_test,
)

automatic_mask = test_margins >= selected_threshold

automatic_count = int(automatic_mask.sum())
review_count = len(test_df) - automatic_count
automation_coverage = automatic_count / len(test_df)

if automatic_count > 0:
    automatic_accuracy = accuracy_score(
        y_test[automatic_mask],
        test_predictions[automatic_mask],
    )

    automatic_macro_f1 = f1_score(
        y_test[automatic_mask],
        test_predictions[automatic_mask],
        average="macro",
        zero_division=0,
    )
else:
    automatic_accuracy = 0
    automatic_macro_f1 = 0

print("\n" + "=" * 65)
print("SELECTIVE PRIORITY RESULTS")
print("=" * 65)
print(
    f"Automatically prioritized: {automatic_count:,}"
)
print(f"Sent for human review: {review_count:,}")
print(
    f"Automation coverage: {automation_coverage * 100:.2f}%"
)
print(
    "Automatic-priority accuracy: "
    f"{automatic_accuracy * 100:.2f}%"
)
print(
    "Automatic-priority macro F1: "
    f"{automatic_macro_f1 * 100:.2f}%"
)

# =========================================================
# Confusion matrix
# =========================================================

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

plt.title("Advanced Priority Model Confusion Matrix")
plt.xlabel("Predicted priority")
plt.ylabel("Actual priority")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "confusion_matrix.png",
    dpi=200,
)

plt.close()

# =========================================================
# Save model package
# =========================================================

model_package = {
    "word_vectorizer": word_vectorizer,
    "character_vectorizer": character_vectorizer,
    "type_encoder": type_encoder,
    "classifier": best_model,
    "model_name": best_name,
    "labels": labels,
    "review_threshold": selected_threshold,
    "subject_repetitions": 3,
    "uses_ticket_type": True,
    "uses_severity_features": True,
}

model_path = MODEL_DIR / "priority_model_v2.joblib"

joblib.dump(model_package, model_path)

summary = {
    "dataset": DATASET_PATH.name,
    "model": best_name,
    "inputs": [
        "subject",
        "body",
        "ticket_type",
        "engineered_severity_features",
    ],
    "training_tickets": len(train_df),
    "validation_tickets": len(validation_df),
    "test_tickets": len(test_df),
    "overall_test_accuracy": round(
        float(test_accuracy),
        6,
    ),
    "test_macro_precision": round(
        float(test_precision),
        6,
    ),
    "test_macro_recall": round(
        float(test_recall),
        6,
    ),
    "test_macro_f1": round(
        float(test_macro_f1),
        6,
    ),
    "test_weighted_f1": round(
        float(test_weighted_f1),
        6,
    ),
    "review_threshold": round(
        float(selected_threshold),
        6,
    ),
    "automatic_priority_accuracy": round(
        float(automatic_accuracy),
        6,
    ),
    "automatic_priority_macro_f1": round(
        float(automatic_macro_f1),
        6,
    ),
    "automation_coverage": round(
        float(automation_coverage),
        6,
    ),
    "sent_for_review": int(review_count),
}

with open(
    MODEL_DIR / "priority_model_v2_info.json",
    "w",
    encoding="utf-8",
) as info_file:
    json.dump(summary, info_file, indent=4)

print("\nMODEL SAVED")
print(model_path)

print("\nREPORTS SAVED")
print(REPORT_DIR)

print(
    "\nUse overall test accuracy and selective accuracy "
    "as two separate claims."
)