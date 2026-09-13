from pathlib import Path
import gc
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "splits"
OUTPUT = ROOT / "reports" / "fast_ensemble"
MODELS = ROOT / "models"

OLD_MODEL = MODELS / "best_it_tuned.joblib"
WINNER_PATH = MODELS / "best_it_fast_comparison.joblib"


def load_data(filename):
    path = SPLITS / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    required = ["ticket_text", "category_id", "group_id"]

    if not set(required).issubset(df.columns):
        raise ValueError(f"Missing required columns in {filename}")

    if df[required].isna().any().any():
        raise ValueError(f"Missing values in {filename}")

    if df["ticket_text"].str.strip().eq("").any():
        raise ValueError(f"Empty ticket text in {filename}")

    return df


def main():
    train = load_data("internal_it_train.csv")
    validation = load_data("internal_it_validation.csv")

    if set(train["group_id"]) & set(validation["group_id"]):
        raise ValueError("Training and validation groups overlap.")

    if not OLD_MODEL.exists():
        raise FileNotFoundError(f"Missing existing model: {OLD_MODEL}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    y_train = train["category_id"]
    y_validation = validation["category_id"]
    labels = sorted(y_train.unique())

    results = []
    best_score = (-1.0, -1.0)
    best_name = None

    def evaluate_and_save(name, pipeline, predictions, seconds):
        nonlocal best_score, best_name

        accuracy = accuracy_score(y_validation, predictions)
        macro_f1 = f1_score(
            y_validation,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        results.append({
            "model": name,
            "validation_accuracy": accuracy,
            "validation_macro_f1": macro_f1,
            "elapsed_seconds": round(seconds, 1),
        })

        print(
            f"{name}\n"
            f"Accuracy: {accuracy:.2%} | Macro F1: {macro_f1:.2%}",
            flush=True,
        )

        # Keep the existing model if every new candidate is worse.
        score = (macro_f1, accuracy)
        if score > best_score:
            best_score = score
            best_name = name
            joblib.dump(pipeline, WINNER_PATH)
            print("Saved current winner.", flush=True)

        pd.DataFrame(results).to_csv(
            OUTPUT / "validation_comparison.csv",
            index=False,
        )

    # 1. Include your original tuned model in the comparison.
    print("\nEvaluating existing model...", flush=True)
    started = time.perf_counter()
    existing = joblib.load(OLD_MODEL)
    predictions = existing.predict(validation["ticket_text"])

    evaluate_and_save(
        "existing_word_character_svm",
        existing,
        predictions,
        time.perf_counter() - started,
    )
    del existing
    gc.collect()

    # 2. Learn new word features from TRAINING data only.
    print("\nPreparing word and phrase features...", flush=True)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2,
        max_features=150000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float64,
    )

    x_train = vectorizer.fit_transform(train["ticket_text"])
    x_validation = vectorizer.transform(validation["ticket_text"])

    print(
        f"Training tickets: {len(train):,}\n"
        f"Validation tickets: {len(validation):,}\n"
        f"Features: {x_train.shape[1]:,}",
        flush=True,
    )

    # A small, fixed set of candidates.
    svm = LinearSVC(
        C=0.5,
        class_weight="balanced",
        max_iter=10000,
        random_state=42,
    )

    logistic = LogisticRegression(
        C=4.0,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
    )

    naive_bayes = ComplementNB(alpha=0.5)

    candidates = [
        ("phrase_svm", clone(svm)),
        ("phrase_logistic_regression", clone(logistic)),
        ("phrase_naive_bayes", clone(naive_bayes)),
        (
            "weighted_hard_voting",
            VotingClassifier(
                estimators=[
                    ("svm", clone(svm)),
                    ("logistic", clone(logistic)),
                    ("nb", clone(naive_bayes)),
                ],
                voting="hard",
                weights=[2, 1, 1],
                n_jobs=1,
            ),
        ),
    ]

    # 3. Train sequentially to limit peak CPU/memory use.
    for number, (name, classifier) in enumerate(candidates, start=1):
        print(
            f"\n[{number}/{len(candidates)}] Training {name}...",
            flush=True,
        )
        started = time.perf_counter()

        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_validation)

        # Store the fitted feature extractor and classifier together.
        fitted_pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ])

        evaluate_and_save(
            name,
            fitted_pipeline,
            predictions,
            time.perf_counter() - started,
        )

    # 4. Generate reports for the selected winner.
    winner = joblib.load(WINNER_PATH)
    predictions = winner.predict(validation["ticket_text"])

    report = classification_report(
        y_validation,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        OUTPUT / "winner_classification_report.csv"
    )

    matrix = confusion_matrix(
        y_validation, predictions, labels=labels
    )
    pd.DataFrame(
        matrix, index=labels, columns=labels
    ).to_csv(OUTPUT / "winner_confusion_matrix.csv")

    errors = validation[
        ["ticket_text", "category_id", "group_id"]
    ].copy()
    errors["predicted_category"] = predictions
    errors = errors.loc[
        errors["category_id"] != errors["predicted_category"]
    ]
    errors.to_csv(OUTPUT / "winner_errors.csv", index=False)

    ranked = pd.DataFrame(results).sort_values(
        ["validation_macro_f1", "validation_accuracy"],
        ascending=False,
    )

    print("\nFINAL VALIDATION COMPARISON")
    print(ranked.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    print(f"\nSelected model: {best_name}")
    print(f"Validation accuracy: {best_score[1]:.2%}")
    print(f"Validation macro F1: {best_score[0]:.2%}")

    print("\nPER-CATEGORY RESULTS")
    print(classification_report(
        y_validation,
        predictions,
        labels=labels,
        digits=3,
        zero_division=0,
    ))

    print(f"\nSelected model saved to: {WINNER_PATH}")
    print(f"Reports saved to: {OUTPUT}")
    print("Original model unchanged. Test files were not loaded.")


if __name__ == "__main__":
    main()