from pathlib import Path
import gc

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "reports" / "combined_v2"
MODELS = ROOT / "models" / "combined_v2"


def load_file(filename):
    path = SPLITS / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def make_features():
    return FeatureUnion([
        (
            "words",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=100000,
                sublinear_tf=True,
            ),
        ),
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
    ])


def scores(actual, predicted, labels):
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual,
            predicted,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
    }


def train_task(task, train, validation, target):
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)

    required = ["ticket_text", "group_id", target]
    for name, data in [("train", train), ("validation", validation)]:
        if data.empty:
            raise ValueError(f"{task}: {name} is empty.")
        if data[required].isna().any().any():
            raise ValueError(f"{task}: missing values in {name}.")
        if data["ticket_text"].str.strip().eq("").any():
            raise ValueError(f"{task}: empty text in {name}.")

    if set(train["group_id"]) & set(validation["group_id"]):
        raise ValueError("Training and validation groups overlap.")

    labels = sorted(train[target].unique())
    if set(validation[target]) - set(labels):
        raise ValueError("Validation contains an unseen target label.")

    print(f"\n{'=' * 55}\nTASK: {task}", flush=True)
    print(
        f"Training rows: {len(train):,} | "
        f"Validation rows: {len(validation):,}",
        flush=True,
    )

    majority = train[target].value_counts().idxmax()
    baseline = scores(
        validation[target], [majority] * len(validation), labels
    )
    print(
        f"Majority baseline accuracy: {baseline['accuracy']:.2%}",
        flush=True,
    )

    print("Building word + character features...", flush=True)
    features = make_features()

    # Fit vocabulary and IDF on training text only.
    x_train = features.fit_transform(train["ticket_text"])
    x_validation = features.transform(validation["ticket_text"])

    print(f"Features: {x_train.shape[1]:,}", flush=True)

    results = []
    best_score = (-1.0, -1.0)
    best_path = MODELS / f"{task}_best.joblib"

    # Six fixed configurations per task.
    for c_value in [0.1, 0.5, 2.0]:
        for weight in [None, "balanced"]:
            print(
                f"\nTraining C={c_value}, class_weight={weight}",
                flush=True,
            )

            classifier = LinearSVC(
                C=c_value,
                class_weight=weight,
                max_iter=10000,
                random_state=42,
            )
            classifier.fit(x_train, train[target])
            predictions = classifier.predict(x_validation)

            overall = scores(validation[target], predictions, labels)

            row = {
                "C": c_value,
                "class_weight": str(weight),
                "accuracy": overall["accuracy"],
                "macro_f1": overall["macro_f1"],
            }

            print(
                f"Overall accuracy: {overall['accuracy']:.2%} | "
                f"Macro F1: {overall['macro_f1']:.2%}",
                flush=True,
            )

            if task == "category":
                for source in ["internal_it", "customer_support"]:
                    mask = validation["source_dataset"].eq(source)
                    source_labels = sorted(
                        train.loc[
                            train["source_dataset"].eq(source), target
                        ].unique()
                    )
                    source_scores = scores(
                        validation.loc[mask, target],
                        predictions[mask.to_numpy()],
                        source_labels,
                    )

                    row[f"{source}_accuracy"] = source_scores["accuracy"]
                    row[f"{source}_macro_f1"] = source_scores["macro_f1"]

                    print(
                        f"{source}: accuracy "
                        f"{source_scores['accuracy']:.2%} | "
                        f"Macro F1 {source_scores['macro_f1']:.2%}",
                        flush=True,
                    )

            results.append(row)

            # Select on all target classes, not just the larger source.
            candidate_score = (
                overall["macro_f1"], overall["accuracy"]
            )

            if candidate_score > best_score:
                best_score = candidate_score
                pipeline = Pipeline([
                    ("features", features),
                    ("classifier", classifier),
                ])
                joblib.dump(pipeline, best_path)
                del pipeline
                print("Saved best candidate for this task.", flush=True)

            pd.DataFrame(results).to_csv(
                OUT / f"{task}_comparison.csv", index=False
            )

    best_model = joblib.load(best_path)
    predictions = best_model.predict(validation["ticket_text"])

    report = classification_report(
        validation[target],
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        OUT / f"{task}_class_report.csv"
    )

    output = validation[
        ["ticket_text", "source_dataset", target]
    ].copy()
    output["prediction"] = predictions
    output.to_csv(
        OUT / f"{task}_validation_predictions.csv", index=False
    )

    print(f"\nFINAL {task.upper()} COMPARISON")
    print(
        pd.DataFrame(results)
        .sort_values(["macro_f1", "accuracy"], ascending=False)
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    print("\nPER-CLASS RESULTS")
    print(classification_report(
        validation[target],
        predictions,
        labels=labels,
        digits=3,
        zero_division=0,
    ))

    print(f"Saved candidate: {best_path}", flush=True)


def main():
    train = load_file("combined_train.csv")
    validation = load_file("combined_validation.csv")

    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    # Category prediction uses both datasets.
    train_task("category", train, validation, "category_id")
    gc.collect()

    # Priority prediction uses only actual, available priority labels.
    priority_train = train.loc[
        train["source_dataset"].eq("customer_support")
        & train["priority"].notna()
    ].copy()

    priority_validation = validation.loc[
        validation["source_dataset"].eq("customer_support")
        & validation["priority"].notna()
    ].copy()

    train_task(
        "priority",
        priority_train,
        priority_validation,
        "priority",
    )

    print("\nEXPERIMENT COMPLETE")
    print(f"Reports: {OUT}")
    print("Original models were not replaced.")
    print("Test files were not loaded.")


if __name__ == "__main__":
    main()