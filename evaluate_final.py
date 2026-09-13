from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_it_tuned.joblib"
SPLIT_DIR = ROOT / "data" / "splits"
OUTPUT_DIR = ROOT / "reports" / "final_test"


def main():
    # Load the selected model without retraining it.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    train = pd.read_csv(SPLIT_DIR / "internal_it_train.csv")
    validation = pd.read_csv(
        SPLIT_DIR / "internal_it_validation.csv"
    )
    test = pd.read_csv(SPLIT_DIR / "internal_it_test.csv")

    required = {"ticket_text", "category_id", "group_id"}

    for name, data in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        if not required.issubset(data.columns):
            raise ValueError(f"Required columns missing in {name}.")

        if data[list(required)].isna().any().any():
            raise ValueError(f"Missing values found in {name}.")

    # Confirm the test groups are separate from development data.
    development_groups = (
        set(train["group_id"]) | set(validation["group_id"])
    )

    if development_groups & set(test["group_id"]):
        raise ValueError("Test groups overlap with development data.")

    model = joblib.load(MODEL_PATH)
    labels = list(model.classes_)

    unknown_labels = set(test["category_id"]) - set(labels)
    if unknown_labels:
        raise ValueError(f"Unknown test categories: {unknown_labels}")

    print("Evaluating the saved model on test tickets...", flush=True)

    actual = test["category_id"]
    predicted = model.predict(test["ticket_text"])

    accuracy = accuracy_score(actual, predicted)
    macro_f1 = f1_score(
        actual, predicted, labels=labels,
        average="macro", zero_division=0,
    )
    weighted_f1 = f1_score(
        actual, predicted, labels=labels,
        average="weighted", zero_division=0,
    )

    # Compare with always predicting the most common training label.
    majority_label = train["category_id"].value_counts().idxmax()
    baseline_accuracy = accuracy_score(
        actual, [majority_label] * len(test)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "test_tickets": len(test),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "majority_baseline_accuracy": baseline_accuracy,
        "model_file": str(MODEL_PATH),
    }

    with open(
        OUTPUT_DIR / "metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    report = classification_report(
        actual, predicted, labels=labels,
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    predictions = test[
        ["ticket_text", "category_id", "group_id"]
    ].copy()
    predictions["predicted_category"] = predicted
    predictions["correct"] = (
        predictions["category_id"]
        == predictions["predicted_category"]
    )
    predictions.to_csv(
        OUTPUT_DIR / "test_predictions.csv", index=False
    )

    matrix = confusion_matrix(actual, predicted, labels=labels)
    short_labels = [
        label.replace("internal_it::", "") for label in labels
    ]

    pd.DataFrame(
        matrix, index=short_labels, columns=short_labels
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=short_labels,
    )
    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=False,
    )
    ax.set_title("Internal IT Category Classifier — Final Test")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nFINAL TEST RESULTS")
    print(f"Test tickets: {len(test):,}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro F1: {macro_f1:.2%}")
    print(f"Weighted F1: {weighted_f1:.2%}")
    print(f"Majority baseline accuracy: {baseline_accuracy:.2%}")

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