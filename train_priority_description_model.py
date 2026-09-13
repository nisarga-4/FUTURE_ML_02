"""Train SupportFlow's hybrid description-only priority model.

The browser supplies one ticket description. During training, each structured
row is converted into a natural-language ticket so the model learns both the
wording and the impact signals. The split occurs before fitting any component.
"""

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


SEED = 42
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models" / "priority_description"
REPORT_DIR = ROOT / "reports" / "priority_description"
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


def locate_dataset():
    preferred = RAW_DIR / "support_ticket_priority_dataset_50k.csv"
    if preferred.exists():
        return preferred
    for path in RAW_DIR.glob("*.csv"):
        columns = set(pd.read_csv(path, nrows=0).columns)
        if {"priority", "customers_affected", "downtime_min"}.issubset(columns):
            return path
    raise FileNotFoundError(
        "Put support_ticket_priority_dataset_50k.csv inside data/raw."
    )


def bucket(value, limits, names):
    return names[int(np.searchsorted(limits, float(value), side="right"))]


def canonical_description(row):
    impact = bucket(
        row.customers_affected, [10, 100, 1000],
        ["tiny", "small", "large", "massive"],
    )
    error = bucket(
        row.error_rate_pct, [5, 25, 60],
        ["low", "moderate", "high", "critical"],
    )
    downtime = bucket(
        row.downtime_min, [1, 30, 120],
        ["none", "short", "long", "critical"],
    )
    return (
        f"{row.product_area} support incident affecting "
        f"{int(row.customers_affected)} customers. Error rate "
        f"{float(row.error_rate_pct):.1f} percent. Service unavailable for "
        f"{int(row.downtime_min)} minutes. Customer tier {row.customer_tier}; "
        f"{row.company_size} {row.industry} company in {row.region}. "
        f"Channel {row.booking_channel}; reporter {row.reported_by_role}. "
        f"Payment impact {int(row.payment_impact_flag)}; security incident "
        f"{int(row.security_incident_flag)}; data loss {int(row.data_loss_flag)}; "
        f"runbook {int(row.has_runbook)}; sentiment {row.customer_sentiment}. "
        f"IMPACT_{impact} ERROR_{error} DOWNTIME_{downtime}"
    )


def make_category_maps(frame):
    result = {}
    for text_column, encoded_column in CATEGORY_PAIRS.items():
        pairs = frame[[text_column, encoded_column]].drop_duplicates()
        conflicts = pairs.groupby(text_column)[encoded_column].nunique()
        if (conflicts > 1).any():
            raise ValueError(f"Conflicting encodings in {text_column}.")
        result[text_column] = {
            str(text).strip().lower(): int(encoded)
            for text, encoded in pairs.itertuples(index=False, name=None)
        }
    return result


def choose_threshold(y_true, probabilities, classes, target=0.98):
    confidence = probabilities.max(axis=1)
    predictions = classes[probabilities.argmax(axis=1)]
    candidates = []
    for threshold in np.unique(np.quantile(confidence, np.linspace(0, 0.995, 300))):
        accepted = confidence >= threshold
        if accepted.sum() < max(100, int(len(y_true) * 0.10)):
            continue
        candidates.append({
            "threshold": float(threshold),
            "coverage": float(accepted.mean()),
            "accuracy": float(accuracy_score(y_true[accepted], predictions[accepted])),
        })
    table = pd.DataFrame(candidates)
    eligible = table[table.accuracy >= target]
    if not eligible.empty:
        selected = eligible.sort_values(["coverage", "accuracy"], ascending=False).iloc[0]
    else:
        selected = table.sort_values(["accuracy", "coverage"], ascending=False).iloc[0]
    return selected, table


def main():
    started = time.time()
    path = locate_dataset()
    data = pd.read_csv(path)
    required = set(FEATURES + ["priority", *CATEGORY_PAIRS.keys()])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if data[FEATURES + ["priority"]].isna().any().any():
        raise ValueError("Missing values exist in required training columns.")

    descriptions = data.apply(canonical_description, axis=1)
    indices = np.arange(len(data))
    train_validation, test = train_test_split(
        indices, test_size=0.15, random_state=SEED,
        stratify=data.priority,
    )
    train, validation = train_test_split(
        train_validation, test_size=3 / 17, random_state=SEED,
        stratify=data.priority.iloc[train_validation],
    )

    structured_model = ExtraTreesClassifier(
        n_estimators=250, min_samples_leaf=2, max_features=0.9,
        class_weight="balanced", n_jobs=-1, random_state=SEED,
    )
    structured_model.fit(data.iloc[train][FEATURES], data.priority.iloc[train])

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=120_000,
        sublinear_tf=True, strip_accents="unicode",
    )
    text_train = vectorizer.fit_transform(descriptions.iloc[train])
    text_model = LogisticRegression(
        C=4.0, max_iter=1000, class_weight="balanced", random_state=SEED,
    )
    text_model.fit(text_train, data.priority.iloc[train])

    classes = np.asarray(structured_model.classes_)
    structured_validation = structured_model.predict_proba(data.iloc[validation][FEATURES])
    text_validation = text_model.predict_proba(vectorizer.transform(descriptions.iloc[validation]))
    y_validation = data.priority.iloc[validation].to_numpy()

    best = None
    for text_weight in [0.10, 0.15, 0.20, 0.25, 0.30]:
        probabilities = (
            (1.0 - text_weight) * structured_validation
            + text_weight * text_validation
        )
        predicted = classes[probabilities.argmax(axis=1)]
        macro_f1 = f1_score(y_validation, predicted, average="macro")
        accuracy = accuracy_score(y_validation, predicted)
        score = macro_f1 + 0.15 * accuracy
        if best is None or score > best[0]:
            best = score, text_weight, probabilities

    _, text_weight, validation_probabilities = best
    threshold, threshold_table = choose_threshold(
        y_validation, validation_probabilities, classes,
    )

    structured_test = structured_model.predict_proba(data.iloc[test][FEATURES])
    text_test = text_model.predict_proba(vectorizer.transform(descriptions.iloc[test]))
    test_probabilities = (
        (1.0 - text_weight) * structured_test + text_weight * text_test
    )
    test_predictions = classes[test_probabilities.argmax(axis=1)]
    y_test = data.priority.iloc[test].to_numpy()
    test_confidence = test_probabilities.max(axis=1)
    automatic = test_confidence >= float(threshold.threshold)

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "test_macro_f1": float(f1_score(y_test, test_predictions, average="macro")),
        "automatic_accuracy": float(accuracy_score(y_test[automatic], test_predictions[automatic])),
        "automation_coverage": float(automatic.mean()),
        "review_threshold": float(threshold.threshold),
        "text_weight": float(text_weight),
    }

    package = {
        "structured_model": structured_model,
        "text_vectorizer": vectorizer,
        "text_model": text_model,
        "text_weight": float(text_weight),
        "features": FEATURES,
        "category_maps": make_category_maps(data),
        "classes": classes,
        "review_threshold": float(threshold.threshold),
    }
    joblib.dump(
        package,
        MODEL_DIR / "priority_description_model.joblib",
        compress=3,
    )
    threshold_table.to_csv(REPORT_DIR / "review_thresholds.csv", index=False)
    (REPORT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, test_predictions, digits=4),
        encoding="utf-8",
    )
    metrics["training_seconds"] = time.time() - started
    (MODEL_DIR / "priority_description_model_info.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8",
    )

    print("\nDESCRIPTION-ONLY PRIORITY MODEL")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"Saved: {MODEL_DIR / 'priority_description_model.joblib'}")


if __name__ == "__main__":
    main()
