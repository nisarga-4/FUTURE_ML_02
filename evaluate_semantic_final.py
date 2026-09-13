from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "models"
    / "semantic_clusters"
    / "best_semantic_cluster_model.joblib"
)

OLD_MODEL_PATH = ROOT / "models" / "best_it_tuned.joblib"
TEST_PATH = ROOT / "data" / "splits" / "internal_it_test.csv"
OUTPUT_DIR = ROOT / "reports" / "semantic_final_test"


def transform_tickets(bundle, texts):
    """Apply the saved training transformations to new tickets."""

    exact = bundle["exact_features"].transform(texts)

    word_features = bundle[
        "semantic_vectorizer"
    ].transform(texts)

    lsa = bundle["svd"].transform(word_features)
    lsa = bundle["normalizer"].transform(lsa)
    lsa = lsa.astype(np.float32)

    similarities = lsa @ bundle["centroids"].T

    semantic = csr_matrix(lsa * 2.0)
    similarities = csr_matrix(
        similarities.astype(np.float32) * 2.0
    )

    features = hstack([
        exact,
        semantic,
        similarities,
    ]).tocsr()

    if bundle["feature_type"] == "tfidf_lsa_clusters":
        clusters = bundle["clusterer"].predict(lsa)

        cluster_features = bundle[
            "cluster_encoder"
        ].transform(clusters.reshape(-1, 1))

        features = hstack([
            features,
            cluster_features * 0.5,
        ]).tocsr()

    elif bundle["feature_type"] == "lsa_semantic":
        features = csr_matrix(lsa)

    return features


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model missing: {MODEL_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test data missing: {TEST_PATH}")

    test = pd.read_csv(TEST_PATH)

    required = ["ticket_text", "category_id", "group_id"]

    if test[required].isna().any().any():
        raise ValueError("Missing test values found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading semantic model...", flush=True)
    bundle = joblib.load(MODEL_PATH)

    print(
        f"Selected feature type: {bundle['feature_type']}",
        flush=True,
    )

    print("Transforming final test tickets...", flush=True)
    x_test = transform_tickets(
        bundle,
        test["ticket_text"],
    )

    actual = test["category_id"]
    predicted = bundle["classifier"].predict(x_test)
    labels = bundle["labels"]

    new_accuracy = accuracy_score(actual, predicted)
    new_macro_f1 = f1_score(
        actual,
        predicted,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    new_weighted_f1 = f1_score(
        actual,
        predicted,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    # Compare with the previously selected model.
    old_model = joblib.load(OLD_MODEL_PATH)
    old_predictions = old_model.predict(test["ticket_text"])

    old_accuracy = accuracy_score(actual, old_predictions)
    old_macro_f1 = f1_score(
        actual,
        old_predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    comparison = pd.DataFrame([
        {
            "model": "original_tfidf_svm",
            "test_accuracy": old_accuracy,
            "test_macro_f1": old_macro_f1,
        },
        {
            "model": "semantic_tfidf_lsa_svm",
            "test_accuracy": new_accuracy,
            "test_macro_f1": new_macro_f1,
        },
    ])

    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    report = classification_report(
        actual,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    short_labels = [
        label.replace("internal_it::", "")
        for label in labels
    ]

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=labels,
    )

    pd.DataFrame(
        matrix,
        index=short_labels,
        columns=short_labels,
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    predictions = test[
        ["ticket_text", "category_id", "group_id"]
    ].copy()

    predictions["predicted_category"] = predicted
    predictions["correct"] = (
        predictions["category_id"]
        == predictions["predicted_category"]
    )

    predictions.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    metrics = {
        "test_tickets": len(test),
        "accuracy": new_accuracy,
        "macro_f1": new_macro_f1,
        "weighted_f1": new_weighted_f1,
        "feature_type": bundle["feature_type"],
    }

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\nFINAL MODEL COMPARISON")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nNEW SEMANTIC MODEL RESULTS")
    print(f"Test tickets: {len(test):,}")
    print(f"Accuracy: {new_accuracy:.2%}")
    print(f"Macro F1: {new_macro_f1:.2%}")
    print(f"Weighted F1: {new_weighted_f1:.2%}")

    print("\nPER-CATEGORY RESULTS")
    print(classification_report(
        actual,
        predicted,
        labels=labels,
        target_names=short_labels,
        digits=3,
        zero_division=0,
    ))

    print(f"\nReports saved in: {OUTPUT_DIR}")
    print("The model was evaluated without retraining.")


if __name__ == "__main__":
    main()