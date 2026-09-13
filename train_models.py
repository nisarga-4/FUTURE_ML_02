from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_DIR = Path(__file__).resolve().parent
SPLIT_DIR = PROJECT_DIR / "data" / "splits"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"


def load_split(filename):
    path = SPLIT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\nRun split_data.py first."
        )

    df = pd.read_csv(path)

    required = ["ticket_text", "category_id", "group_id"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"Required columns missing in {filename}")

    if df[required].isna().any().any():
        raise ValueError(f"Missing text, labels, or groups in {filename}")

    return df


def evaluate(model, data, labels):
    predictions = model.predict(data["ticket_text"])

    return {
        "accuracy": accuracy_score(data["category_id"], predictions),
        # Fixed labels keep the IT-only and combined comparison fair.
        "macro_f1": f1_score(
            data["category_id"],
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
    }


def main():
    it_train = load_split("internal_it_train.csv")
    combined_train = load_split("combined_train.csv")
    it_validation = load_split("internal_it_validation.csv")
    customer_validation = load_split("customer_support_validation.csv")

    # Verify training does not contain validation text groups.
    validation_groups = (
        set(it_validation["group_id"])
        | set(customer_validation["group_id"])
    )

    if set(combined_train["group_id"]) & validation_groups:
        raise ValueError("Training and validation groups overlap.")

    # Verify both experiments use the same internal IT training rows.
    combined_it = combined_train.loc[
        combined_train["source_dataset"].eq("internal_it")
    ]
    columns = ["ticket_text", "category_id", "group_id"]

    def comparable(data):
        return (
            data[columns]
            .sort_values(columns)
            .reset_index(drop=True)
        )

    if not comparable(it_train).equals(comparable(combined_it)):
        raise ValueError("Internal IT training rows do not match.")

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    it_labels = sorted(it_train["category_id"].unique())
    customer_labels = sorted(
        combined_train.loc[
            combined_train["source_dataset"].eq("customer_support"),
            "category_id",
        ].unique()
    )

    algorithms = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "naive_bayes": ComplementNB(alpha=0.5),
        "logistic_regression": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
        ),
        "linear_svm": LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=42,
        ),
    }

    experiments = {
        "it_only": it_train,
        "combined": combined_train,
    }

    results = []

    for experiment_name, training_data in experiments.items():
        for algorithm_name, estimator in algorithms.items():
            print(
                f"\nTraining: {experiment_name} / {algorithm_name}",
                flush=True,
            )
            started = time.perf_counter()

            # TF-IDF learns its vocabulary ONLY from training text.
            # Keep negation words such as "not" and "cannot".
            pipeline = Pipeline([
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=100000,
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", clone(estimator)),
            ])

            pipeline.fit(
                training_data["ticket_text"],
                training_data["category_id"],
            )

            it_scores = evaluate(pipeline, it_validation, it_labels)

            row = {
                "experiment": experiment_name,
                "algorithm": algorithm_name,
                "it_validation_accuracy": it_scores["accuracy"],
                "it_validation_macro_f1": it_scores["macro_f1"],
            }

            print(
                f"IT validation accuracy: {it_scores['accuracy']:.2%}\n"
                f"IT validation macro F1: {it_scores['macro_f1']:.2%}",
                flush=True,
            )

            if experiment_name == "combined":
                customer_scores = evaluate(
                    pipeline,
                    customer_validation,
                    customer_labels,
                )

                row["customer_validation_accuracy"] = (
                    customer_scores["accuracy"]
                )
                row["customer_validation_macro_f1"] = (
                    customer_scores["macro_f1"]
                )

                print(
                    "Customer validation accuracy: "
                    f"{customer_scores['accuracy']:.2%}\n"
                    "Customer validation macro F1: "
                    f"{customer_scores['macro_f1']:.2%}",
                    flush=True,
                )

            model_path = (
                MODEL_DIR / f"{experiment_name}_{algorithm_name}.joblib"
            )
            joblib.dump(pipeline, model_path)

            row["elapsed_seconds"] = round(
                time.perf_counter() - started, 1
            )
            results.append(row)

            # Save progress after every completed model.
            pd.DataFrame(results).to_csv(
                REPORT_DIR / "validation_comparison.csv",
                index=False,
            )

            print(
                f"Saved model. Elapsed: {row['elapsed_seconds']} seconds",
                flush=True,
            )

    comparison = pd.DataFrame(results).sort_values(
        "it_validation_macro_f1",
        ascending=False,
    )

    print("\nCOMPARISON ON THE SAME INTERNAL IT VALIDATION TICKETS")
    display_columns = [
        "experiment",
        "algorithm",
        "it_validation_accuracy",
        "it_validation_macro_f1",
    ]
    print(
        comparison[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nTraining complete. Test files were not loaded.")
    print(f"Models: {MODEL_DIR}")
    print(f"Results: {REPORT_DIR / 'validation_comparison.csv'}")


if __name__ == "__main__":
    main()