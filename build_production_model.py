from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "splits"

SOURCE_MODEL = (
    ROOT
    / "models"
    / "semantic_clusters"
    / "best_semantic_cluster_model.joblib"
)

OUTPUT_MODEL = (
    ROOT
    / "models"
    / "production_ticket_model.joblib"
)

METADATA_PATH = (
    ROOT
    / "models"
    / "production_ticket_model_info.json"
)


def create_centroids(lsa, targets, labels):
    targets = np.asarray(targets)
    centroids = []

    for label in labels:
        centroid = lsa[targets == label].mean(axis=0)
        length = np.linalg.norm(centroid)

        if length > 0:
            centroid = centroid / length

        centroids.append(centroid)

    return np.asarray(centroids, dtype=np.float32)


def main():
    print("Loading training and validation tickets...", flush=True)

    train = pd.read_csv(SPLITS / "internal_it_train.csv")
    validation = pd.read_csv(
        SPLITS / "internal_it_validation.csv"
    )

    development = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    if development[
        ["ticket_text", "category_id", "group_id"]
    ].isna().any().any():
        raise ValueError("Missing values found.")

    if development["group_id"].duplicated().any():
        print(
            "Some rows share a group inside development data. "
            "This is allowed during final production training."
        )

    print(f"Production training tickets: {len(development):,}")
    print("Loading semantic feature system...", flush=True)

    bundle = joblib.load(SOURCE_MODEL)
    texts = development["ticket_text"]
    targets = development["category_id"]
    labels = bundle["labels"]

    print("Creating exact TF-IDF features...", flush=True)
    exact = bundle["exact_features"].transform(texts)

    print("Creating LSA semantic features...", flush=True)
    word_features = bundle[
        "semantic_vectorizer"
    ].transform(texts)

    lsa = bundle["svd"].transform(word_features)
    lsa = bundle["normalizer"].transform(lsa)
    lsa = lsa.astype(np.float32)

    # Recalculate category centres using all development tickets.
    centroids = create_centroids(
        lsa,
        targets,
        labels,
    )

    similarities = lsa @ centroids.T

    production_features = hstack([
        exact,
        csr_matrix(lsa * 2.0),
        csr_matrix(
            similarities.astype(np.float32) * 2.0
        ),
    ]).tocsr()

    print(
        f"Total production features: "
        f"{production_features.shape[1]:,}",
        flush=True,
    )

    print("Training final production classifier...", flush=True)

    classifier = LinearSVC(
        C=0.2,
        class_weight="balanced",
        max_iter=10000,
        random_state=42,
    )

    classifier.fit(production_features, targets)

    # Update only the components changed during final training.
    bundle["centroids"] = centroids
    bundle["classifier"] = classifier
    bundle["feature_type"] = "tfidf_lsa"
    bundle["training_rows"] = len(development)
    bundle["production_model"] = True

    joblib.dump(bundle, OUTPUT_MODEL)

    metadata = {
        "model": "TF-IDF + LSA + Linear SVM",
        "training_tickets": len(development),
        "categories": len(labels),
        "C": 0.2,
        "class_weight": "balanced",
        "reference_test_accuracy": 0.8725,
        "reference_test_macro_f1": 0.8730,
        "note": (
            "Reference metrics belong to the selected model trained "
            "before adding validation data."
        ),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nPRODUCTION MODEL COMPLETE")
    print(f"Training tickets used: {len(development):,}")
    print(f"Categories: {len(labels)}")
    print(f"Saved model: {OUTPUT_MODEL}")
    print(f"Saved information: {METADATA_PATH}")
    print("\nReference test accuracy: 87.25%")
    print(
        "The production model used additional validation data, "
        "so it needs fresh external tickets for a new accuracy claim."
    )


if __name__ == "__main__":
    main()