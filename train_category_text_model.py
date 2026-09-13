"""Train the final subject + description SupportFlow category model."""

from pathlib import Path
from itertools import product
import json
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


SEED = 42
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models" / "category_text"
REPORT_DIR = ROOT / "reports" / "category_text"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Harmonizes the customer dataset's 16 issue subjects into the router's six
# operational categories. This is an explicit business taxonomy, not a model
# feature and not a copy of the router label.
CUSTOMER_SUBJECT_MAP = {
    "Payment issue": "billing",
    "Refund request": "billing",
    "Cancellation request": "cancellation",
    "Software bug": "technical",
    "Hardware issue": "technical",
    "Battery life": "technical",
    "Network problem": "technical",
    "Installation support": "technical",
    "Product setup": "technical",
    "Account access": "technical",
    "Peripheral compatibility": "technical",
    "Data loss": "technical",
    "Display issue": "technical",
    "Product compatibility": "upgrade",
    "Product recommendation": "upgrade",
    "Delivery problem": "complaint",
}


def locate(name):
    candidates = [RAW_DIR / name, ROOT / name]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Put {name} inside data/raw.")


def normalize(value):
    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+\.\S+", " email ", text)
    text = re.sub(r"\b\d{7,}\b", " number ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_category_data():
    router = pd.read_csv(locate("support_ticket_router_12k.csv"))
    customer = pd.read_csv(locate("customer_support_tickets.csv"))

    router_rows = pd.DataFrame({
        "text": router["text"].fillna("").astype(str),
        "label": router["label"].fillna("").astype(str).str.lower().str.strip(),
        "source": "router_12k",
    })
    customer_rows = pd.DataFrame({
        "text": (
            customer["Ticket Subject"].fillna("").astype(str)
            + ". "
            + customer["Ticket Description"].fillna("").astype(str)
        ),
        "label": customer["Ticket Subject"].map(CUSTOMER_SUBJECT_MAP),
        "source": "customer_support",
    })
    data = pd.concat([router_rows, customer_rows], ignore_index=True).dropna()
    data["normalized"] = data.text.map(normalize)
    data = data[(data.normalized != "") & (data.label != "")]

    conflicts = data.groupby("normalized").label.nunique()
    conflicted = set(conflicts[conflicts > 1].index)
    data = data[~data.normalized.isin(conflicted)]
    return data.drop_duplicates("normalized").reset_index(drop=True)


def operational_billing_examples():
    """Domain adaptation examples added to training only, never validation/test."""
    services = [
        "payment service", "checkout system", "billing platform",
        "transaction processor", "invoice service",
    ]
    states = [
        "is unavailable", "is down in production", "is failing",
        "has stopped working", "returns errors",
    ]
    impacts = [
        "customers cannot complete checkout", "transactions are failing",
        "payments cannot be processed", "orders cannot be paid",
        "billing operations are blocked",
    ]
    affected = [
        "500 customers", "over 1000 users", "many enterprise customers",
        "all customers", "250 accounts",
    ]
    examples = []
    for service, state, impact, users in product(services, states, impacts, affected):
        examples.append(
            f"The {service} {state}. {users} are affected and {impact}. "
            "This is a production outage."
        )
        if len(examples) >= 600:
            break
    return examples


def explicit_cancellation_examples():
    """Train the model to prioritize explicit cancellation intent over billing words."""
    openings = [
        "Please close our account", "Please cancel our subscription",
        "We want to terminate the service", "Deactivate our company account",
        "End our membership", "Discontinue our current plan",
    ]
    timings = [
        "at the end of this month", "when the current billing period finishes",
        "before the next renewal date", "effective immediately",
        "after our current contract ends",
    ]
    reasons = [
        "We do not want another renewal charge",
        "We have exported all required information",
        "The service is no longer required",
        "Do not renew the plan automatically",
        "Confirm that no future invoices will be created",
    ]
    return [
        f"{opening} {timing}. {reason}."
        for opening, timing, reason in product(openings, timings, reasons)
    ]


def choose_threshold(y_true, probabilities, classes, target=0.98):
    confidence = probabilities.max(axis=1)
    predictions = classes[probabilities.argmax(axis=1)]
    rows = []
    for threshold in np.unique(np.quantile(confidence, np.linspace(0, 0.995, 300))):
        accepted = confidence >= threshold
        if accepted.sum() < max(100, int(len(y_true) * 0.10)):
            continue
        rows.append({
            "threshold": float(threshold),
            "coverage": float(accepted.mean()),
            "accuracy": float(accuracy_score(y_true[accepted], predictions[accepted])),
        })
    table = pd.DataFrame(rows)
    eligible = table[table.accuracy >= target]
    selected = (
        eligible.sort_values(["coverage", "accuracy"], ascending=False).iloc[0]
        if not eligible.empty
        else table.sort_values(["accuracy", "coverage"], ascending=False).iloc[0]
    )
    return selected, table


def main():
    started = time.time()
    data = load_category_data()
    indices = np.arange(len(data))
    train_validation, test = train_test_split(
        indices, test_size=0.15, random_state=SEED, stratify=data.label,
    )
    train, validation = train_test_split(
        train_validation, test_size=3 / 17, random_state=SEED,
        stratify=data.label.iloc[train_validation],
    )

    billing_text = operational_billing_examples()
    cancellation_text = explicit_cancellation_examples()
    adapted_text = billing_text + cancellation_text
    adapted_labels = (
        ["billing"] * len(billing_text)
        + ["cancellation"] * len(cancellation_text)
    )
    training_text = pd.concat(
        [data.text.iloc[train], pd.Series(adapted_text)], ignore_index=True,
    )
    training_labels = pd.concat(
        [data.label.iloc[train], pd.Series(adapted_labels)],
        ignore_index=True,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.995,
        max_features=180_000, sublinear_tf=True, strip_accents="unicode",
    )
    x_train = vectorizer.fit_transform(training_text)
    x_validation = vectorizer.transform(data.text.iloc[validation])
    x_test = vectorizer.transform(data.text.iloc[test])

    best = None
    for c_value in [0.20, 0.35, 0.50, 0.75, 1.00]:
        candidate = LinearSVC(
            C=c_value, class_weight="balanced", random_state=SEED,
        )
        candidate.fit(x_train, training_labels)
        prediction = candidate.predict(x_validation)
        macro_f1 = f1_score(data.label.iloc[validation], prediction, average="macro")
        accuracy = accuracy_score(data.label.iloc[validation], prediction)
        score = macro_f1 + 0.15 * accuracy
        if best is None or score > best[0]:
            best = score, c_value

    _, c_value = best
    model = CalibratedClassifierCV(
        LinearSVC(C=c_value, class_weight="balanced", random_state=SEED),
        method="sigmoid", cv=5, n_jobs=-1,
    )
    model.fit(x_train, training_labels)

    validation_probabilities = model.predict_proba(x_validation)
    threshold, threshold_table = choose_threshold(
        data.label.iloc[validation].to_numpy(),
        validation_probabilities,
        model.classes_,
    )
    test_probabilities = model.predict_proba(x_test)
    test_predictions = model.classes_[test_probabilities.argmax(axis=1)]
    y_test = data.label.iloc[test].to_numpy()
    automatic = test_probabilities.max(axis=1) >= float(threshold.threshold)

    metrics = {
        "rows_after_cleaning": int(len(data)),
        "training_rows": int(len(train)),
        "training_only_domain_examples": int(len(adapted_text)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "selected_C": float(c_value),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "test_macro_f1": float(f1_score(y_test, test_predictions, average="macro")),
        "automatic_accuracy": float(accuracy_score(y_test[automatic], test_predictions[automatic])),
        "automation_coverage": float(automatic.mean()),
        "review_threshold": float(threshold.threshold),
        "training_seconds": time.time() - started,
    }
    package = {
        "vectorizer": vectorizer,
        "model": model,
        "classes": model.classes_,
        "review_threshold": float(threshold.threshold),
        "taxonomy": sorted(model.classes_.tolist()),
    }
    joblib.dump(package, MODEL_DIR / "category_text_model.joblib", compress=3)
    threshold_table.to_csv(REPORT_DIR / "review_thresholds.csv", index=False)
    (REPORT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, test_predictions, digits=4), encoding="utf-8",
    )
    (MODEL_DIR / "category_text_model_info.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8",
    )

    print("\nFINAL SUBJECT + DESCRIPTION CATEGORY MODEL")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print(f"Saved: {MODEL_DIR / 'category_text_model.joblib'}")


if __name__ == "__main__":
    main()
