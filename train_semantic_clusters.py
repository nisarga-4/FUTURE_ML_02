from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "splits"
MODELS = ROOT / "models" / "semantic_clusters"
REPORTS = ROOT / "reports" / "semantic_clusters"

EXISTING_MODEL = ROOT / "models" / "best_it_tuned.joblib"


def load_data(filename):
    path = SPLITS / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\nRun split_data.py first."
        )

    data = pd.read_csv(path)
    required = ["ticket_text", "category_id", "group_id"]

    if not set(required).issubset(data.columns):
        raise ValueError(f"Missing columns in {filename}")

    if data[required].isna().any().any():
        raise ValueError(f"Missing values in {filename}")

    return data


def calculate_scores(actual, predicted, labels):
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


def create_class_centroids(semantic_features, labels, class_names):
    """Create one semantic centre for every ticket category."""
    centroids = []

    labels_array = np.asarray(labels)

    for class_name in class_names:
        class_rows = semantic_features[
            labels_array == class_name
        ]

        centroid = class_rows.mean(axis=0)
        norm = np.linalg.norm(centroid)

        if norm > 0:
            centroid = centroid / norm

        centroids.append(centroid)

    return np.asarray(centroids, dtype=np.float32)


def main():
    train = load_data("internal_it_train.csv")
    validation = load_data("internal_it_validation.csv")

    if set(train["group_id"]) & set(validation["group_id"]):
        raise ValueError("Training and validation groups overlap.")

    if not EXISTING_MODEL.exists():
        raise FileNotFoundError(
            f"Existing model missing: {EXISTING_MODEL}"
        )

    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    train_text = train["ticket_text"]
    validation_text = validation["ticket_text"]

    y_train = train["category_id"]
    y_validation = validation["category_id"]
    labels = sorted(y_train.unique())

    results = []

    # ---------------------------------------------------------
    # 1. Evaluate existing model
    # ---------------------------------------------------------
    print("\nEvaluating existing TF-IDF model...", flush=True)

    existing_model = joblib.load(EXISTING_MODEL)
    existing_predictions = existing_model.predict(validation_text)

    existing_scores = calculate_scores(
        y_validation,
        existing_predictions,
        labels,
    )

    results.append({
        "model": "existing_word_character_svm",
        "C": 0.5,
        "validation_accuracy": existing_scores["accuracy"],
        "validation_macro_f1": existing_scores["macro_f1"],
    })

    print(
        f"Accuracy: {existing_scores['accuracy']:.2%}\n"
        f"Macro F1: {existing_scores['macro_f1']:.2%}",
        flush=True,
    )

    # ---------------------------------------------------------
    # 2. Exact word and character features
    # ---------------------------------------------------------
    print("\nCreating exact word and character features...", flush=True)

    exact_features = FeatureUnion([
        (
            "words",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=100000,
                sublinear_tf=True,
                dtype=np.float32,
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
                dtype=np.float32,
            ),
        ),
    ])

    x_exact_train = exact_features.fit_transform(train_text)
    x_exact_validation = exact_features.transform(validation_text)

    print(
        f"Exact feature count: {x_exact_train.shape[1]:,}",
        flush=True,
    )

    # ---------------------------------------------------------
    # 3. LSA semantic features
    # ---------------------------------------------------------
    print("\nCreating LSA semantic features...", flush=True)

    semantic_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=60000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    x_word_train = semantic_vectorizer.fit_transform(train_text)
    x_word_validation = semantic_vectorizer.transform(
        validation_text
    )

    # LSA compresses thousands of word features into 200
    # broader semantic dimensions.
    svd = TruncatedSVD(
        n_components=200,
        n_iter=7,
        random_state=42,
    )

    normalizer = Normalizer(copy=False)

    x_lsa_train = svd.fit_transform(x_word_train)
    x_lsa_train = normalizer.fit_transform(x_lsa_train)

    x_lsa_validation = svd.transform(x_word_validation)
    x_lsa_validation = normalizer.transform(x_lsa_validation)

    x_lsa_train = x_lsa_train.astype(np.float32)
    x_lsa_validation = x_lsa_validation.astype(np.float32)

    explained = svd.explained_variance_ratio_.sum()

    print(
        f"Semantic dimensions: {x_lsa_train.shape[1]}\n"
        f"Explained variance: {explained:.2%}",
        flush=True,
    )

    # ---------------------------------------------------------
    # 4. Class-centroid similarity features
    # ---------------------------------------------------------
    centroids = create_class_centroids(
        x_lsa_train,
        y_train,
        labels,
    )

    # Each ticket receives eight similarity scores:
    # one score for each known category centre.
    train_similarities = x_lsa_train @ centroids.T
    validation_similarities = x_lsa_validation @ centroids.T

    train_similarities = csr_matrix(
        train_similarities.astype(np.float32) * 2.0
    )
    validation_similarities = csr_matrix(
        validation_similarities.astype(np.float32) * 2.0
    )

    # ---------------------------------------------------------
    # 5. Semantic clustering
    # ---------------------------------------------------------
    print("\nFinding semantic ticket clusters...", flush=True)

    clusterer = MiniBatchKMeans(
        n_clusters=64,
        batch_size=2048,
        n_init=10,
        max_iter=200,
        random_state=42,
    )

    train_clusters = clusterer.fit_predict(x_lsa_train)
    validation_clusters = clusterer.predict(x_lsa_validation)

    cluster_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )

    x_cluster_train = cluster_encoder.fit_transform(
        train_clusters.reshape(-1, 1)
    )
    x_cluster_validation = cluster_encoder.transform(
        validation_clusters.reshape(-1, 1)
    )

    # Create a report showing how clean each cluster is.
    cluster_data = pd.DataFrame({
        "cluster": train_clusters,
        "category": y_train.to_numpy(),
    })

    cluster_report = (
        cluster_data.groupby("cluster")["category"]
        .value_counts()
        .rename("count")
        .reset_index()
    )

    totals = (
        cluster_report.groupby("cluster")["count"]
        .sum()
        .rename("cluster_size")
    )

    dominant = (
        cluster_report.sort_values(
            ["cluster", "count"],
            ascending=[True, False],
        )
        .drop_duplicates("cluster")
        .set_index("cluster")
    )

    summary = dominant[["category", "count"]].copy()
    summary.columns = ["dominant_category", "dominant_count"]
    summary["cluster_size"] = totals
    summary["purity"] = (
        summary["dominant_count"] / summary["cluster_size"]
    )

    summary.sort_values("purity").to_csv(
        REPORTS / "cluster_summary.csv"
    )

    # Save examples from mixed clusters for later inspection.
    low_purity_clusters = summary.loc[
        summary["purity"] < 0.70
    ].index

    mixed_examples = train.loc[
        np.isin(train_clusters, low_purity_clusters),
        ["ticket_text", "category_id", "group_id"],
    ].copy()

    mixed_examples["cluster"] = train_clusters[
        np.isin(train_clusters, low_purity_clusters)
    ]

    mixed_examples.to_csv(
        REPORTS / "mixed_cluster_examples.csv",
        index=False,
    )

    print(
        f"Clusters created: {len(summary)}\n"
        f"Clusters below 70% purity: "
        f"{(summary['purity'] < 0.70).sum()}",
        flush=True,
    )

    # ---------------------------------------------------------
    # 6. Construct candidate feature sets
    # ---------------------------------------------------------
    semantic_train = csr_matrix(x_lsa_train * 2.0)
    semantic_validation = csr_matrix(x_lsa_validation * 2.0)

    hybrid_semantic_train = hstack([
        x_exact_train,
        semantic_train,
        train_similarities,
    ]).tocsr()

    hybrid_semantic_validation = hstack([
        x_exact_validation,
        semantic_validation,
        validation_similarities,
    ]).tocsr()

    hybrid_cluster_train = hstack([
        hybrid_semantic_train,
        x_cluster_train * 0.5,
    ]).tocsr()

    hybrid_cluster_validation = hstack([
        hybrid_semantic_validation,
        x_cluster_validation * 0.5,
    ]).tocsr()

    feature_sets = {
        "lsa_semantic": (
            csr_matrix(x_lsa_train),
            csr_matrix(x_lsa_validation),
        ),
        "tfidf_lsa": (
            hybrid_semantic_train,
            hybrid_semantic_validation,
        ),
        "tfidf_lsa_clusters": (
            hybrid_cluster_train,
            hybrid_cluster_validation,
        ),
    }

    best_new_score = (-1.0, -1.0)
    best_new_details = None
    best_classifier = None

    total_runs = len(feature_sets) * 3
    run_number = 0

    # ---------------------------------------------------------
    # 7. Train the classifiers
    # ---------------------------------------------------------
    for feature_name, (
        x_train,
        x_validation,
    ) in feature_sets.items():

        for c_value in [0.2, 0.5, 1.0]:
            run_number += 1
            started = time.perf_counter()

            print(
                f"\n[{run_number}/{total_runs}] "
                f"{feature_name}, C={c_value}",
                flush=True,
            )

            classifier = LinearSVC(
                C=c_value,
                class_weight="balanced",
                max_iter=10000,
                random_state=42,
            )

            classifier.fit(x_train, y_train)
            predictions = classifier.predict(x_validation)

            score = calculate_scores(
                y_validation,
                predictions,
                labels,
            )

            row = {
                "model": feature_name,
                "C": c_value,
                "validation_accuracy": score["accuracy"],
                "validation_macro_f1": score["macro_f1"],
                "elapsed_seconds": round(
                    time.perf_counter() - started,
                    1,
                ),
            }

            results.append(row)

            print(
                f"Accuracy: {score['accuracy']:.2%} | "
                f"Macro F1: {score['macro_f1']:.2%}",
                flush=True,
            )

            candidate_score = (
                score["macro_f1"],
                score["accuracy"],
            )

            if candidate_score > best_new_score:
                best_new_score = candidate_score
                best_new_details = row.copy()
                best_classifier = classifier

                joblib.dump(
                    {
                        "exact_features": exact_features,
                        "semantic_vectorizer": semantic_vectorizer,
                        "svd": svd,
                        "normalizer": normalizer,
                        "centroids": centroids,
                        "clusterer": clusterer,
                        "cluster_encoder": cluster_encoder,
                        "feature_type": feature_name,
                        "classifier": classifier,
                        "labels": labels,
                    },
                    MODELS / "best_semantic_cluster_model.joblib",
                )

                print("Saved new semantic candidate.", flush=True)

            pd.DataFrame(results).to_csv(
                REPORTS / "validation_comparison.csv",
                index=False,
            )

    # ---------------------------------------------------------
    # 8. Final validation report for best new candidate
    # ---------------------------------------------------------
    best_feature_name = best_new_details["model"]
    best_validation_features = feature_sets[
        best_feature_name
    ][1]

    best_predictions = best_classifier.predict(
        best_validation_features
    )

    report = classification_report(
        y_validation,
        best_predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORTS / "best_semantic_classification_report.csv"
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

    print("\nBEST NEW SEMANTIC SETTINGS")
    print(best_new_details)

    if best_new_score > (
        existing_scores["macro_f1"],
        existing_scores["accuracy"],
    ):
        print("\nRESULT: The semantic model improved validation performance.")
    else:
        print(
            "\nRESULT: The existing TF-IDF model remains stronger."
        )

    print(
        f"\nExisting accuracy: "
        f"{existing_scores['accuracy']:.2%}"
    )
    print(
        f"Best semantic accuracy: "
        f"{best_new_details['validation_accuracy']:.2%}"
    )
    print(f"\nReports saved in: {REPORTS}")
    print(f"Semantic model saved in: {MODELS}")
    print("Test files were not loaded.")


if __name__ == "__main__":
    main()