from pathlib import Path
import gc
import time

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "splits"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports" / "tuning"


def load_data(filename):
    path = SPLITS / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    data = pd.read_csv(path)
    required = ["ticket_text", "category_id", "group_id"]

    if not set(required).issubset(data.columns):
        raise ValueError(f"Missing required columns in {filename}")

    if data[required].isna().any().any():
        raise ValueError(f"Missing values found in {filename}")

    return data


def main():
    train = load_data("internal_it_train.csv")
    validation = load_data("internal_it_validation.csv")

    if set(train["group_id"]) & set(validation["group_id"]):
        raise ValueError("Training and validation groups overlap.")

    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    labels = sorted(train["category_id"].unique())
    y_train = train["category_id"]
    y_validation = validation["category_id"]

    word_features = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
    )

    feature_options = {
        # Matches the previous IT SVM's feature settings.
        "words": word_features,

        "words_and_characters": FeatureUnion([
            ("words", clone(word_features)),
            (
                "characters",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=100000,
                    sublinear_tf=True,
                ),
            ),
        ]),
    }

    # A bounded search: 2 feature choices x 3 C values x 2 weights.
    c_values = [0.5, 1.0, 2.0]
    weights = [None, "balanced"]

    total_runs = len(feature_options) * len(c_values) * len(weights)
    run = 0
    results = []
    best_score = (-1.0, -1.0)
    best_details = None

    best_path = MODELS / "best_it_tuned.joblib"

    for feature_name, feature_template in feature_options.items():
        print(f"\nPreparing features: {feature_name}", flush=True)
        features = clone(feature_template)

        # Learn vocabulary and IDF from TRAINING data only.
        x_train = features.fit_transform(train["ticket_text"])
        x_validation = features.transform(validation["ticket_text"])

        print(
            f"Feature count: {x_train.shape[1]:,}",
            flush=True,
        )

        for c_value in c_values:
            for weight in weights:
                run += 1
                started = time.perf_counter()

                print(
                    f"\n[{run}/{total_runs}] {feature_name}, "
                    f"C={c_value}, class_weight={weight}",
                    flush=True,
                )

                classifier = LinearSVC(
                    C=c_value,
                    class_weight=weight,
                    max_iter=10000,
                    random_state=42,
                )
                classifier.fit(x_train, y_train)

                predictions = classifier.predict(x_validation)

                accuracy = accuracy_score(y_validation, predictions)
                macro_f1 = f1_score(
                    y_validation,
                    predictions,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )

                details = {
                    "features": feature_name,
                    "C": c_value,
                    "class_weight": str(weight),
                    "validation_accuracy": accuracy,
                    "validation_macro_f1": macro_f1,
                    "training_seconds": round(
                        time.perf_counter() - started, 1
                    ),
                }
                results.append(details)

                print(
                    f"Accuracy: {accuracy:.2%} | "
                    f"Macro F1: {macro_f1:.2%}",
                    flush=True,
                )

                # Macro F1 first; accuracy breaks an exact tie.
                score = (macro_f1, accuracy)

                if score > best_score:
                    best_score = score
                    best_details = details.copy()

                    # Both steps are already fitted.
                    fitted_pipeline = Pipeline([
                        ("features", features),
                        ("classifier", classifier),
                    ])
                    joblib.dump(fitted_pipeline, best_path)

                    print("Saved new best model.", flush=True)
                    del fitted_pipeline

                # Keep completed results even if a later run stops.
                pd.DataFrame(results).to_csv(
                    REPORTS / "tuning_results.csv",
                    index=False,
                )

        del x_train, x_validation, features, classifier
        gc.collect()

    # Inspect the saved winner on validation data only.
    best_model = joblib.load(best_path)
    predictions = best_model.predict(validation["ticket_text"])

    report = classification_report(
        y_validation,
        predictions,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(
        REPORTS / "validation_class_report.csv"
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=labels,
    )
    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    ).to_csv(REPORTS / "validation_confusion_matrix.csv")

    errors = validation[
        ["ticket_text", "category_id", "group_id"]
    ].copy()
    errors["predicted_category"] = predictions
    errors = errors.loc[
        errors["category_id"] != errors["predicted_category"]
    ]
    errors.to_csv(
        REPORTS / "validation_errors.csv",
        index=False,
    )

    ranked = pd.DataFrame(results).sort_values(
        ["validation_macro_f1", "validation_accuracy"],
        ascending=False,
    )

    print("\nFINAL VALIDATION COMPARISON")
    print(
        ranked.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nBEST SETTINGS")
    print(best_details)

    print("\nPER-CATEGORY VALIDATION RESULTS")
    print(
        classification_report(
            y_validation,
            predictions,
            labels=labels,
            digits=3,
            zero_division=0,
        )
    )

    print(f"\nBest model: {best_path}")
    print(f"Reports: {REPORTS}")
    print("Test data was not loaded.")


if __name__ == "__main__":
    main()