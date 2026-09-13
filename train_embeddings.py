from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models" / "embedding_experiment"
REPORT_DIR = ROOT / "reports" / "embedding_experiment"
CACHE_DIR = ROOT / "data" / "embedding_cache"

ENCODER_NAME = "sentence-transformers/all-mpnet-base-v2"


def load_split(filename):
    path = SPLITS / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    required = ["ticket_text", "category_id", "group_id"]

    if not set(required).issubset(df.columns):
        raise ValueError(f"Missing required columns in {filename}")

    if df[required].isna().any().any():
        raise ValueError(f"Missing values in {filename}")

    return df


def get_embeddings(encoder, texts, split_name):
    # Include text order and encoder settings in the cache identity.
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "encoder": ENCODER_NAME,
                "max_length": encoder.max_seq_length,
                "normalize": True,
                "texts": texts,
            },
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:20]

    path = CACHE_DIR / f"{split_name}_{fingerprint}.npy"

    if path.exists():
        print(f"Loading cached {split_name} embeddings.", flush=True)
        values = np.load(path, allow_pickle=False)
        if len(values) != len(texts):
            raise ValueError("Cached embedding row count is incorrect.")
        return values

    print(f"Encoding {len(texts):,} {split_name} tickets...", flush=True)

    values = encoder.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Finish writing before replacing the cache file.
    temporary = path.with_suffix(".tmp")
    with open(temporary, "wb") as file:
        np.save(file, values)
    temporary.replace(path)

    return values


def main():
    train = load_split("internal_it_train.csv")
    validation = load_split("internal_it_validation.csv")

    if set(train["group_id"]) & set(validation["group_id"]):
        raise ValueError("Training and validation groups overlap.")

    baseline_path = ROOT / "models" / "best_it_tuned.joblib"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing existing model: {baseline_path}")

    for folder in [MODEL_DIR, REPORT_DIR, CACHE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    labels = sorted(train["category_id"].unique())
    y_train = train["category_id"]
    y_validation = validation["category_id"]

    # Compare with the actual saved TF-IDF model.
    baseline = joblib.load(baseline_path)
    baseline_predictions = baseline.predict(validation["ticket_text"])

    baseline_accuracy = accuracy_score(
        y_validation, baseline_predictions
    )
    baseline_f1 = f1_score(
        y_validation,
        baseline_predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    print(
        f"\nExisting TF-IDF validation accuracy: {baseline_accuracy:.2%}"
        f"\nExisting TF-IDF validation macro F1: {baseline_f1:.2%}",
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nEncoding device: {device}", flush=True)
    print("Loading encoder; the first run needs internet.", flush=True)

    local_encoder = MODEL_DIR / "encoder"
    encoder = SentenceTransformer(
        str(local_encoder) if local_encoder.exists() else ENCODER_NAME,
        device=device,
    )

    if not local_encoder.exists():
        encoder.save(str(local_encoder))

    print(
        f"Encoder maximum input length: "
        f"{encoder.max_seq_length} tokens",
        flush=True,
    )

    x_train = get_embeddings(
        encoder, train["ticket_text"].tolist(), "train"
    )
    x_validation = get_embeddings(
        encoder, validation["ticket_text"].tolist(), "validation"
    )

    best_score = (-1.0, -1.0)
    results = []
    best_path = MODEL_DIR / "best_classifier.joblib"

    # Fixed, small experiment: six configurations.
    for c_value in [0.5, 2.0, 8.0]:
        for weight in [None, "balanced"]:
            print(
                f"\nTraining SVM: C={c_value}, class_weight={weight}",
                flush=True,
            )

            classifier = LinearSVC(
                C=c_value,
                class_weight=weight,
                dual="auto",
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

            row = {
                "C": c_value,
                "class_weight": str(weight),
                "validation_accuracy": accuracy,
                "validation_macro_f1": macro_f1,
            }
            results.append(row)

            print(
                f"Accuracy: {accuracy:.2%} | Macro F1: {macro_f1:.2%}",
                flush=True,
            )

            if (macro_f1, accuracy) > best_score:
                best_score = (macro_f1, accuracy)
                joblib.dump(classifier, best_path)

                metadata = {
                    **row,
                    "encoder": ENCODER_NAME,
                    "normalize_embeddings": True,
                    "max_seq_length": encoder.max_seq_length,
                }
                with open(
                    MODEL_DIR / "settings.json",
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(metadata, file, indent=2)

                print("Saved best embedding classifier.", flush=True)

            pd.DataFrame(results).to_csv(
                REPORT_DIR / "validation_results.csv",
                index=False,
            )

    best_classifier = joblib.load(best_path)
    predictions = best_classifier.predict(x_validation)

    report = classification_report(
        y_validation,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        REPORT_DIR / "classification_report.csv"
    )

    comparison = pd.DataFrame([
        {
            "model": "Existing TF-IDF + SVM",
            "accuracy": baseline_accuracy,
            "macro_f1": baseline_f1,
        },
        {
            "model": "Embeddings + SVM",
            "accuracy": best_score[1],
            "macro_f1": best_score[0],
        },
    ])

    comparison.to_csv(REPORT_DIR / "comparison.csv", index=False)

    print("\nFINAL VALIDATION COMPARISON")
    print(comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    print("\nEMBEDDING MODEL PER-CATEGORY RESULTS")
    print(classification_report(
        y_validation,
        predictions,
        labels=labels,
        digits=3,
        zero_division=0,
    ))

    print(f"\nNew model components: {MODEL_DIR}")
    print(f"Reports: {REPORT_DIR}")
    print("Your existing TF-IDF model was not changed.")
    print("Test files were not loaded.")


if __name__ == "__main__":
    main()