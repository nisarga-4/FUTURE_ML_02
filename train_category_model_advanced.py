"""Advanced category model: TF-IDF SVM + bagged semantic neural models.

All model/blend/threshold choices use training and validation data. The test
partition is transformed only after those choices have been fixed.
"""

from pathlib import Path
import json
import re
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import softmax
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
# A slightly stricter validation target provides a buffer for unseen tickets.
TARGET_ROUTING_ACCURACY = 0.93
PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
SPLIT_DIR = PROJECT_DIR / "data" / "splits"
MODEL_DIR = PROJECT_DIR / "models" / "category_advanced"
REPORT_DIR = PROJECT_DIR / "reports" / "category_advanced"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(value):
    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+\.\S+", " email ", text)
    text = re.sub(r"\b\d{7,}\b", " number ", text)
    return re.sub(r"\s+", " ", text).strip()


def locate_raw_dataset():
    preferred = RAW_DIR / "all_tickets_processed_improved_v3.csv"
    if preferred.exists():
        return preferred
    for path in RAW_DIR.glob("*.csv"):
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
            if {"Document", "Topic_group"}.issubset(columns):
                return path
        except Exception:
            continue
    raise FileNotFoundError(
        "Dataset with Document and Topic_group was not found in data/raw."
    )


def standardize(frame):
    text_column = next(
        (
            x
            for x in ["ticket_text", "text", "Document", "document"]
            if x in frame.columns
        ),
        None,
    )
    label_column = next(
        (x for x in ["category", "label", "Topic_group", "topic_group"]
         if x in frame.columns), None
    )
    if text_column is None or label_column is None:
        raise ValueError(
            "Could not identify the ticket text and category columns. "
            f"Available columns: {frame.columns.tolist()}"
        )
    result = frame[[text_column, label_column]].rename(
        columns={text_column: "Document", label_column: "Topic_group"}
    )
    result.Document = result.Document.fillna("").astype(str).map(normalize_text)
    result.Topic_group = result.Topic_group.fillna("").astype(str).str.strip()
    return result[(result.Document != "") & (result.Topic_group != "")].reset_index(drop=True)


def find_split(kind):
    if not SPLIT_DIR.exists():
        return None
    exact = SPLIT_DIR / f"internal_it_{kind}.csv"
    if exact.exists():
        return exact
    matches = sorted(SPLIT_DIR.glob(f"*internal*{kind}*.csv"))
    return matches[0] if matches else None


def load_data():
    paths = {kind: find_split(kind) for kind in ["train", "validation", "test"]}
    if all(paths.values()):
        print("Using existing leakage-safe internal-IT split files.")
        train = standardize(pd.read_csv(paths["train"]))
        validation = standardize(pd.read_csv(paths["validation"]))
        test = standardize(pd.read_csv(paths["test"]))
        overlaps = [
            len(set(train.Document) & set(validation.Document)),
            len(set(train.Document) & set(test.Document)),
            len(set(validation.Document) & set(test.Document)),
        ]
        if any(overlaps):
            raise ValueError(f"Text leakage detected between splits: {overlaps}")
        return train, validation, test, locate_raw_dataset().name

    dataset_path = locate_raw_dataset()
    print(f"Creating a reproducible split from {dataset_path.name}.")
    data = standardize(pd.read_csv(dataset_path))
    conflicts = data.groupby("Document").Topic_group.nunique()
    bad = set(conflicts[conflicts > 1].index)
    data = data[~data.Document.isin(bad)].drop_duplicates("Document").reset_index(drop=True)
    train_validation, test = train_test_split(
        data, test_size=0.20, random_state=RANDOM_STATE,
        stratify=data.Topic_group,
    )
    train, validation = train_test_split(
        train_validation, test_size=0.125, random_state=RANDOM_STATE,
        stratify=train_validation.Topic_group,
    )
    return train, validation, test, dataset_path.name


def class_centroids(word_features, labels, number_of_classes):
    values = []
    for index in range(number_of_classes):
        values.append(np.asarray(word_features[labels == index].mean(axis=0)).ravel())
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def normalized_svm_probabilities(scores):
    centered = scores - scores.mean(axis=1, keepdims=True)
    scaled = centered / np.maximum(scores.std(axis=1, keepdims=True), 1e-6)
    return softmax(scaled, axis=1)


def tempered_probabilities(probabilities, temperature):
    return softmax(
        np.log(np.maximum(probabilities, 1e-8)) / temperature, axis=1
    )


def metric_objective(actual, scores):
    predicted = scores.argmax(axis=1)
    macro = f1_score(actual, predicted, average="macro")
    accuracy = accuracy_score(actual, predicted)
    return macro + 0.15 * accuracy


def calibrate_class_biases(actual, scores):
    bias = np.zeros(scores.shape[1], dtype=np.float32)
    best = metric_objective(actual, scores + bias)
    for step in [0.01, 0.005, 0.002, 0.001]:
        improved = True
        while improved:
            improved = False
            for category in range(len(bias)):
                for direction in [-1, 1]:
                    candidate = bias.copy()
                    candidate[category] += direction * step
                    value = metric_objective(actual, scores + candidate)
                    if value > best + 1e-8:
                        bias, best, improved = candidate, value, True
    return bias


def choose_review_threshold(actual, predictions, margins):
    rows = []
    for threshold in np.unique(np.quantile(margins, np.linspace(0, 0.995, 500))):
        accepted = margins >= threshold
        if accepted.sum() < max(100, int(len(actual) * 0.10)):
            continue
        rows.append({
            "threshold": float(threshold),
            "coverage": float(accepted.mean()),
            "automatic_accuracy": accuracy_score(actual[accepted], predictions[accepted]),
            "automatic_tickets": int(accepted.sum()),
        })
    table = pd.DataFrame(rows)
    eligible = table[table.automatic_accuracy >= TARGET_ROUTING_ACCURACY]
    selected = (
        eligible.sort_values(["coverage", "automatic_accuracy"], ascending=False).iloc[0]
        if not eligible.empty
        else table.sort_values(["automatic_accuracy", "coverage"], ascending=False).iloc[0]
    )
    return float(selected.threshold), selected, table


started = time.time()
train, validation, test, dataset_name = load_data()
print(f"Training tickets: {len(train):,}")
print(f"Validation tickets: {len(validation):,}")
print(f"Untouched test tickets: {len(test):,}")

encoder = LabelEncoder()
y_train = encoder.fit_transform(train.Topic_group)
y_validation = encoder.transform(validation.Topic_group)
number_of_classes = len(encoder.classes_)

print("\nBuilding word TF-IDF...")
word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), min_df=2, max_df=0.995, max_features=220_000,
    sublinear_tf=True, strip_accents="unicode", dtype=np.float32,
)
xw_train = word_vectorizer.fit_transform(train.Document)
xw_validation = word_vectorizer.transform(validation.Document)

print("Building character TF-IDF...")
character_vectorizer = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000,
    sublinear_tf=True, strip_accents="unicode", dtype=np.float32,
)
xc_train = character_vectorizer.fit_transform(train.Document)
xc_validation = character_vectorizer.transform(validation.Document)
exact_train = hstack([xw_train, xc_train], format="csr")
exact_validation = hstack([xw_validation, xc_validation], format="csr")

print("Training supervised-similarity SVM...")
centroids = class_centroids(xw_train, y_train, number_of_classes)
augmented_train = hstack(
    [exact_train, csr_matrix(xw_train @ centroids.T) * 4.0], format="csr"
)
augmented_validation = hstack(
    [exact_validation, csr_matrix(xw_validation @ centroids.T) * 4.0],
    format="csr",
)
svm = LinearSVC(C=0.16, class_weight="balanced", random_state=RANDOM_STATE)
svm.fit(augmented_train, y_train)
validation_svm_probability = normalized_svm_probabilities(
    svm.decision_function(augmented_validation)
)

print("Creating 500 semantic dimensions...")
svd = TruncatedSVD(n_components=500, n_iter=7, random_state=RANDOM_STATE)
normalizer = Normalizer(copy=False)
semantic_train = normalizer.fit_transform(svd.fit_transform(xw_train)).astype(np.float32)
semantic_validation = normalizer.transform(svd.transform(xw_validation)).astype(np.float32)
print(f"Semantic explained variance: {svd.explained_variance_ratio_.sum() * 100:.2f}%")

mlp_configurations = [
    ((256,), 0.0001, 42),
    ((256,), 0.0001, 123),
    ((512,), 0.0003, 777),
]
neural_models = []
validation_neural_probabilities = []
for number, (hidden, alpha, seed) in enumerate(mlp_configurations, 1):
    print(f"Training semantic neural model {number}/{len(mlp_configurations)}...")
    model = MLPClassifier(
        hidden_layer_sizes=hidden, activation="relu", alpha=alpha,
        batch_size=256, learning_rate_init=0.001, max_iter=100,
        early_stopping=True, validation_fraction=0.10, n_iter_no_change=7,
        random_state=seed,
    )
    model.fit(semantic_train, y_train)
    neural_models.append(model)
    validation_neural_probabilities.append(model.predict_proba(semantic_validation))

print("Selecting the blend using validation data...")
best = None
for model_count in range(1, len(neural_models) + 1):
    average_neural = np.mean(
        validation_neural_probabilities[:model_count], axis=0
    )
    for temperature in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        neural_probability = tempered_probabilities(average_neural, temperature)
        for neural_weight in np.arange(0.0, 0.501, 0.025):
            combined = (
                (1.0 - neural_weight) * validation_svm_probability
                + neural_weight * neural_probability
            )
            candidate = {
                "objective": metric_objective(y_validation, combined),
                "model_count": model_count,
                "temperature": temperature,
                "neural_weight": float(neural_weight),
                "scores": combined,
            }
            if best is None or candidate["objective"] > best["objective"]:
                best = candidate

class_biases = calibrate_class_biases(y_validation, best["scores"])
validation_scores = best["scores"] + class_biases
validation_predictions = validation_scores.argmax(axis=1)
validation_accuracy = accuracy_score(y_validation, validation_predictions)
validation_macro_f1 = f1_score(y_validation, validation_predictions, average="macro")
print(f"Selected neural models: {best['model_count']}")
print(f"Selected neural weight: {best['neural_weight']:.3f}")
print(f"Selected temperature: {best['temperature']:.2f}")
print(f"Validation accuracy: {validation_accuracy * 100:.2f}%")
print(f"Validation macro F1: {validation_macro_f1 * 100:.2f}%")

validation_order = np.sort(validation_scores, axis=1)
validation_margins = validation_order[:, -1] - validation_order[:, -2]
review_threshold, threshold_row, threshold_table = choose_review_threshold(
    y_validation, validation_predictions, validation_margins
)
threshold_table.to_csv(REPORT_DIR / "review_thresholds.csv", index=False)
print(f"Review threshold: {review_threshold:.4f}")
print(f"Validation routing accuracy: {threshold_row.automatic_accuracy * 100:.2f}%")
print(f"Validation routing coverage: {threshold_row.coverage * 100:.2f}%")

# No test transformation occurred before this point.
print("\nEvaluating the fixed system once on the test tickets...")
y_test = encoder.transform(test.Topic_group)
xw_test = word_vectorizer.transform(test.Document)
xc_test = character_vectorizer.transform(test.Document)
exact_test = hstack([xw_test, xc_test], format="csr")
augmented_test = hstack(
    [exact_test, csr_matrix(xw_test @ centroids.T) * 4.0], format="csr"
)
test_svm_probability = normalized_svm_probabilities(
    svm.decision_function(augmented_test)
)
semantic_test = normalizer.transform(svd.transform(xw_test)).astype(np.float32)
average_neural_test = np.mean(
    [model.predict_proba(semantic_test)
     for model in neural_models[:best["model_count"]]],
    axis=0,
)
test_neural_probability = tempered_probabilities(
    average_neural_test, best["temperature"]
)
test_scores = (
    (1.0 - best["neural_weight"]) * test_svm_probability
    + best["neural_weight"] * test_neural_probability
    + class_biases
)
test_predictions = test_scores.argmax(axis=1)
test_order = np.sort(test_scores, axis=1)
test_margins = test_order[:, -1] - test_order[:, -2]

accuracy = accuracy_score(y_test, test_predictions)
macro_f1 = f1_score(y_test, test_predictions, average="macro")
weighted_f1 = f1_score(y_test, test_predictions, average="weighted")
macro_precision = precision_score(y_test, test_predictions, average="macro")
macro_recall = recall_score(y_test, test_predictions, average="macro")
automatic = test_margins >= review_threshold
routing_accuracy = accuracy_score(y_test[automatic], test_predictions[automatic])
coverage = float(automatic.mean())

print("\nFINAL ADVANCED CATEGORY RESULTS")
print(f"Overall test accuracy: {accuracy * 100:.2f}%")
print(f"Macro precision: {macro_precision * 100:.2f}%")
print(f"Macro recall: {macro_recall * 100:.2f}%")
print(f"Macro F1: {macro_f1 * 100:.2f}%")
print(f"Weighted F1: {weighted_f1 * 100:.2f}%")
print(f"Automatic-routing accuracy: {routing_accuracy * 100:.2f}%")
print(f"Automation coverage: {coverage * 100:.2f}%")
print(f"Tickets sent for review: {(~automatic).sum():,}")

labels = np.arange(number_of_classes)
report = classification_report(
    y_test, test_predictions, labels=labels,
    target_names=encoder.classes_, digits=3, zero_division=0,
)
print("\nPER-CATEGORY RESULTS\n" + report)
(REPORT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
matrix = confusion_matrix(y_test, test_predictions, labels=labels)
pd.DataFrame(matrix, index=encoder.classes_, columns=encoder.classes_).to_csv(
    REPORT_DIR / "confusion_matrix.csv"
)
plt.figure(figsize=(11, 8))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Advanced Category Model Confusion Matrix")
plt.xlabel("Predicted category")
plt.ylabel("Actual category")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=200)
plt.close()

package = {
    "word_vectorizer": word_vectorizer,
    "character_vectorizer": character_vectorizer,
    "centroids": centroids,
    "centroid_weight": 4.0,
    "svm": svm,
    "svd": svd,
    "normalizer": normalizer,
    "neural_models": neural_models[:best["model_count"]],
    "temperature": best["temperature"],
    "neural_weight": best["neural_weight"],
    "class_biases": class_biases,
    "classes": encoder.classes_,
    "review_threshold": review_threshold,
}
model_path = MODEL_DIR / "category_model_advanced.joblib"
joblib.dump(package, model_path, compress=3)
summary = {
    "dataset": dataset_name,
    "train_tickets": len(train),
    "validation_tickets": len(validation),
    "test_tickets": len(test),
    "overall_test_accuracy": round(float(accuracy), 6),
    "test_macro_precision": round(float(macro_precision), 6),
    "test_macro_recall": round(float(macro_recall), 6),
    "test_macro_f1": round(float(macro_f1), 6),
    "test_weighted_f1": round(float(weighted_f1), 6),
    "automatic_routing_accuracy": round(float(routing_accuracy), 6),
    "automation_coverage": round(coverage, 6),
    "review_threshold": round(float(review_threshold), 6),
    "neural_models": best["model_count"],
    "neural_weight": best["neural_weight"],
    "temperature": best["temperature"],
    "elapsed_seconds": round(time.time() - started, 1),
}
(MODEL_DIR / "category_model_advanced_info.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(f"\nSaved model: {model_path}")
print(f"Saved reports: {REPORT_DIR}")
print(f"Elapsed: {time.time() - started:.1f} seconds")
