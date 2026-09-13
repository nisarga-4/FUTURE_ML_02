from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "models"
    / "semantic_clusters"
    / "best_semantic_cluster_model.joblib"
)

VALIDATION_PATH = (
    ROOT / "data" / "splits" / "internal_it_validation.csv"
)

TEST_PATH = (
    ROOT / "data" / "splits" / "internal_it_test.csv"
)

REPORT_DIR = ROOT / "reports" / "review_calibration"
THRESHOLD_PATH = ROOT / "models" / "review_threshold.json"

# We target 92% on validation to provide some safety
# when checking the untouched test tickets.
TARGET_VALIDATION_ACCURACY = 0.92
MINIMUM_ACCEPTED_TICKETS = 300


def transform_tickets(bundle, texts):
    exact = bundle["exact_features"].transform(texts)

    word_features = bundle[
        "semantic_vectorizer"
    ].transform(texts)

    lsa = bundle["svd"].transform(word_features)
    lsa = bundle["normalizer"].transform(lsa)
    lsa = lsa.astype(np.float32)

    similarities = lsa @ bundle["centroids"].T

    return hstack([
        exact,
        csr_matrix(lsa * 2.0),
        csr_matrix(
            similarities.astype(np.float32) * 2.0
        ),
    ]).tocsr()


def predict_with_margins(bundle, data):
    features = transform_tickets(
        bundle,
        data["ticket_text"],
    )

    classifier = bundle["classifier"]
    predictions = classifier.predict(features)
    scores = classifier.decision_function(features)

    ordered_scores = np.sort(scores, axis=1)
    margins = ordered_scores[:, -1] - ordered_scores[:, -2]

    return predictions, margins


def find_threshold(actual, predicted, margins):
    correct = np.asarray(actual) == np.asarray(predicted)

    # Test every possible margin boundary.
    candidates = np.unique(margins)
    best = None

    for threshold in candidates:
        accepted = margins >= threshold
        accepted_count = accepted.sum()

        if accepted_count < MINIMUM_ACCEPTED_TICKETS:
            continue

        accepted_accuracy = correct[accepted].mean()
        coverage = accepted.mean()

        if accepted_accuracy >= TARGET_VALIDATION_ACCURACY:
            if best is None or coverage > best["coverage"]:
                best = {
                    "threshold": float(threshold),
                    "accuracy": float(accepted_accuracy),
                    "coverage": float(coverage),
                    "accepted": int(accepted_count),
                    "reviewed": int((~accepted).sum()),
                }

    if best is None:
        raise RuntimeError(
            "No threshold reached the target with enough tickets."
        )

    return best


def evaluate_selective(
    name,
    actual,
    predicted,
    margins,
    threshold,
):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    accepted = margins >= threshold
    reviewed = ~accepted

    automatic_accuracy = accuracy_score(
        actual[accepted],
        predicted[accepted],
    )

    result = {
        "dataset": name,
        "total_tickets": int(len(actual)),
        "automatically_routed": int(accepted.sum()),
        "sent_for_review": int(reviewed.sum()),
        "automation_coverage": float(accepted.mean()),
        "automatic_routing_accuracy": float(
            automatic_accuracy
        ),
        "overall_model_accuracy": float(
            accuracy_score(actual, predicted)
        ),
    }

    return result, accepted


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(MODEL_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    test = pd.read_csv(TEST_PATH)

    print("Calculating validation decision margins...", flush=True)

    validation_predictions, validation_margins = (
        predict_with_margins(bundle, validation)
    )

    selected = find_threshold(
        validation["category_id"],
        validation_predictions,
        validation_margins,
    )

    threshold = selected["threshold"]

    print(f"Selected decision threshold: {threshold:.4f}")
    print(
        "Validation automatic-routing accuracy: "
        f"{selected['accuracy']:.2%}"
    )
    print(
        f"Validation coverage: {selected['coverage']:.2%}"
    )

    print("\nAuditing the threshold on test tickets...", flush=True)

    test_predictions, test_margins = predict_with_margins(
        bundle,
        test,
    )

    validation_result, validation_accepted = (
        evaluate_selective(
            "validation",
            validation["category_id"],
            validation_predictions,
            validation_margins,
            threshold,
        )
    )

    test_result, test_accepted = evaluate_selective(
        "test",
        test["category_id"],
        test_predictions,
        test_margins,
        threshold,
    )

    results = pd.DataFrame([
        validation_result,
        test_result,
    ])

    results.to_csv(
        REPORT_DIR / "selective_routing_results.csv",
        index=False,
    )

    test_output = test[
        ["ticket_text", "category_id", "group_id"]
    ].copy()

    test_output["predicted_category"] = test_predictions
    test_output["decision_margin"] = test_margins
    test_output["routing_decision"] = np.where(
        test_accepted,
        "Automatic",
        "Manual review",
    )
    test_output["correct"] = (
        test_output["category_id"]
        == test_output["predicted_category"]
    )

    test_output.to_csv(
        REPORT_DIR / "test_routing_decisions.csv",
        index=False,
    )

    threshold_information = {
        "decision_margin_threshold": threshold,
        "target_validation_accuracy": (
            TARGET_VALIDATION_ACCURACY
        ),
        "validation_coverage": selected["coverage"],
        "test_automatic_routing_accuracy": (
            test_result["automatic_routing_accuracy"]
        ),
        "test_automation_coverage": (
            test_result["automation_coverage"]
        ),
        "model": str(MODEL_PATH),
    }

    with open(
        THRESHOLD_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(threshold_information, file, indent=2)

    print("\nSELECTIVE ROUTING RESULTS")
    print(results.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    print(
        "\nTest automatic-routing accuracy: "
        f"{test_result['automatic_routing_accuracy']:.2%}"
    )
    print(
        "Test automation coverage: "
        f"{test_result['automation_coverage']:.2%}"
    )
    print(
        "Tickets sent for review: "
        f"{test_result['sent_for_review']:,}"
    )

    print(f"\nSaved threshold: {THRESHOLD_PATH}")
    print(f"Reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()