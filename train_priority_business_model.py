"""Train a structured business-priority model on the 50K ticket dataset.

This intentionally excludes priority_cat (a direct copy of the target),
ticket_id, company_id, and duplicated text/category representations.
"""

from pathlib import Path
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
PROJECT = Path(__file__).resolve().parent
RAW_DIR = PROJECT / "data" / "raw"
MODEL_DIR = PROJECT / "models" / "priority_business"
REPORT_DIR = PROJECT / "reports" / "priority_business"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "day_of_week_num", "company_size_cat", "industry_cat",
    "customer_tier_cat", "org_users", "region_cat", "past_30d_tickets",
    "past_90d_incidents", "product_area_cat", "booking_channel_cat",
    "reported_by_role_cat", "customers_affected", "error_rate_pct",
    "downtime_min", "payment_impact_flag", "security_incident_flag",
    "data_loss_flag", "has_runbook", "customer_sentiment_cat",
    "description_length",
]

CATEGORY_PAIRS = {
    "company_size": "company_size_cat",
    "industry": "industry_cat",
    "customer_tier": "customer_tier_cat",
    "region": "region_cat",
    "product_area": "product_area_cat",
    "booking_channel": "booking_channel_cat",
    "reported_by_role": "reported_by_role_cat",
    "customer_sentiment": "customer_sentiment_cat",
}

FORBIDDEN = {
    "priority", "priority_cat", "ticket_id", "company_id",
    *CATEGORY_PAIRS.keys(),
}


def locate_dataset():
    preferred = [
        "support_ticket_priority_dataset_50k.csv",
        "support-ticket-priority-dataset-50k.csv",
        "priority_dataset_50k.csv",
    ]
    for name in preferred:
        path = RAW_DIR / name
        if path.exists():
            return path
    for path in RAW_DIR.glob("*.csv"):
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
            if {"priority", "customers_affected", "downtime_min", "priority_cat"}.issubset(columns):
                return path
        except Exception:
            continue
    raise FileNotFoundError(
        "The 50K priority CSV was not found. Put it inside data/raw and rerun."
    )


def category_maps(frame):
    maps = {}
    for text_column, number_column in CATEGORY_PAIRS.items():
        pairs = frame[[text_column, number_column]].drop_duplicates()
        if text_column == "customer_sentiment":
            pairs[text_column] = pairs[text_column].fillna("unknown")
        conflicts = pairs.groupby(text_column)[number_column].nunique()
        if (conflicts > 1).any():
            raise ValueError(f"Non-unique category mapping for {text_column}")
        maps[text_column] = {
            str(row[text_column]).lower(): int(row[number_column])
            for _, row in pairs.iterrows()
        }
    return maps


def select_review_threshold(actual, probabilities, target=0.99):
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    rows = []
    for threshold in np.unique(np.quantile(confidence, np.linspace(0, 0.995, 500))):
        accepted = confidence >= threshold
        if accepted.sum() < max(100, int(len(actual) * 0.10)):
            continue
        rows.append({
            "threshold": float(threshold),
            "coverage": float(accepted.mean()),
            "accuracy": accuracy_score(actual[accepted], predicted[accepted]),
            "tickets": int(accepted.sum()),
        })
    table = pd.DataFrame(rows)
    eligible = table[table.accuracy >= target]
    selected = (
        eligible.sort_values(["coverage", "accuracy"], ascending=False).iloc[0]
        if not eligible.empty
        else table.sort_values(["accuracy", "coverage"], ascending=False).iloc[0]
    )
    return selected, table


started = time.time()
dataset_path = locate_dataset()
data = pd.read_csv(dataset_path)
print(f"Loading: {dataset_path.name}")
print(f"Rows: {len(data):,}")

missing = set(FEATURES + ["priority"]) - set(data.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")
if set(FEATURES) & FORBIDDEN:
    raise RuntimeError("A forbidden leakage column entered the feature list.")
if data.ticket_id.duplicated().any():
    raise ValueError("Duplicate ticket IDs detected; inspect before training.")
if data[FEATURES].duplicated().any():
    raise ValueError("Duplicate complete feature rows detected; inspect before splitting.")
if data[FEATURES + ["priority"]].isna().any().any():
    raise ValueError("Unexpected missing values in model columns.")

print("Leakage columns excluded: priority_cat, ticket_id, company_id")
print("Priority distribution:")
print(data.priority.value_counts())

train_validation, test = train_test_split(
    data, test_size=0.15, random_state=SEED, stratify=data.priority
)
train, validation = train_test_split(
    train_validation, test_size=3 / 17,
    random_state=SEED, stratify=train_validation.priority,
)
X_train, y_train = train[FEATURES], train.priority
X_validation, y_validation = validation[FEATURES], validation.priority
print("\nDATA SPLIT")
print(f"Training: {len(train):,}")
print(f"Validation: {len(validation):,}")
print(f"Untouched test: {len(test):,}")

models = {
    "majority_baseline": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, max_features=0.8,
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
    ),
    "extra_trees": ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=2, max_features=0.9,
        class_weight="balanced", n_jobs=-1, random_state=SEED,
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_leaf_nodes=15, learning_rate=0.08, max_iter=300,
        min_samples_leaf=20, l2_regularization=1.0, random_state=SEED,
    ),
}

rows = []
winner = None
for name, model in models.items():
    model_started = time.time()
    print(f"\nTraining {name}...", flush=True)
    model.fit(X_train, y_train)
    prediction = model.predict(X_validation)
    accuracy = accuracy_score(y_validation, prediction)
    macro_f1 = f1_score(y_validation, prediction, average="macro")
    row = {
        "model": name,
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1,
        "seconds": time.time() - model_started,
    }
    rows.append(row)
    print(f"Accuracy: {accuracy:.2%} | Macro F1: {macro_f1:.2%}")
    score = macro_f1 + 0.15 * accuracy
    if winner is None or score > winner[0]:
        winner = (score, name, model)

_, winner_name, winner_model = winner
comparison = pd.DataFrame(rows).sort_values(
    ["validation_macro_f1", "validation_accuracy"], ascending=False
)
print("\nVALIDATION COMPARISON")
print(comparison.to_string(index=False))
print(f"\nSelected model: {winner_name}")

classes = np.asarray(winner_model.classes_)
validation_probabilities = winner_model.predict_proba(X_validation)
y_validation_number = np.searchsorted(classes, y_validation.to_numpy())
selected_threshold, threshold_table = select_review_threshold(
    y_validation_number, validation_probabilities
)

# Test data is first used here after model and threshold selection are fixed.
print("\nEvaluating the fixed model on untouched test tickets...")
X_test, y_test = test[FEATURES], test.priority
test_prediction = winner_model.predict(X_test)
test_probabilities = winner_model.predict_proba(X_test)
test_accuracy = accuracy_score(y_test, test_prediction)
test_macro_f1 = f1_score(y_test, test_prediction, average="macro")
test_weighted_f1 = f1_score(y_test, test_prediction, average="weighted")
confidence = test_probabilities.max(axis=1)
automatic = confidence >= float(selected_threshold.threshold)
automatic_accuracy = accuracy_score(y_test[automatic], test_prediction[automatic])

print("\n" + "=" * 64)
print("FINAL BUSINESS PRIORITY TEST RESULTS")
print("=" * 64)
print(f"Test tickets: {len(test):,}")
print(f"Overall accuracy: {test_accuracy:.2%}")
print(f"Macro F1: {test_macro_f1:.2%}")
print(f"Weighted F1: {test_weighted_f1:.2%}")
print(f"Automatic-priority accuracy: {automatic_accuracy:.2%}")
print(f"Automation coverage: {automatic.mean():.2%}")
print(f"Sent for review: {(~automatic).sum():,}")

report = classification_report(y_test, test_prediction, digits=3)
print("\nPER-PRIORITY RESULTS")
print(report)

importance = permutation_importance(
    winner_model, X_validation, y_validation,
    scoring="accuracy", n_repeats=5, random_state=SEED, n_jobs=-1,
)
importance_table = pd.DataFrame({
    "feature": FEATURES,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std,
}).sort_values("importance_mean", ascending=False)
print("\nTOP FEATURES")
print(importance_table.head(12).to_string(index=False))

comparison.to_csv(REPORT_DIR / "validation_comparison.csv", index=False)
threshold_table.to_csv(REPORT_DIR / "review_thresholds.csv", index=False)
importance_table.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
(REPORT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
matrix = confusion_matrix(y_test, test_prediction, labels=classes)
pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
    REPORT_DIR / "confusion_matrix.csv"
)
plt.figure(figsize=(7, 5))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted priority")
plt.ylabel("Actual priority")
plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=180)
plt.close()

package = {
    "model": winner_model,
    "model_name": winner_name,
    "features": FEATURES,
    "category_maps": category_maps(data),
    "classes": classes,
    "review_threshold": float(selected_threshold.threshold),
}
model_path = MODEL_DIR / "priority_business_model.joblib"
joblib.dump(package, model_path, compress=3)
information = {
    "dataset": dataset_path.name,
    "training_tickets": len(train),
    "validation_tickets": len(validation),
    "test_tickets": len(test),
    "selected_model": winner_name,
    "test_accuracy": test_accuracy,
    "test_macro_f1": test_macro_f1,
    "test_weighted_f1": test_weighted_f1,
    "automatic_priority_accuracy": automatic_accuracy,
    "automation_coverage": float(automatic.mean()),
    "review_threshold": float(selected_threshold.threshold),
    "elapsed_seconds": time.time() - started,
    "excluded_columns": sorted(FORBIDDEN),
}
(MODEL_DIR / "priority_business_model_info.json").write_text(
    json.dumps(information, indent=2), encoding="utf-8"
)
print(f"\nSaved model: {model_path}")
print(f"Saved reports: {REPORT_DIR}")
print(f"Elapsed: {time.time() - started:.1f} seconds")
