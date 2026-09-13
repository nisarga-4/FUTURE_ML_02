"""Predict ticket category with category_model_advanced.joblib."""

from pathlib import Path
import re

import joblib
import numpy as np
from scipy.special import softmax
from scipy.sparse import csr_matrix, hstack

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_DIR / "models" / "category_advanced" / "category_model_advanced.joblib"
)

TEAM_MAPPING = {
    "Access": "Identity and Access Team",
    "Administrative rights": "System Administration Team",
    "HR Support": "HR Technology Team",
    "Hardware": "Hardware Support Team",
    "Internal Project": "Internal Projects Team",
    "Miscellaneous": "General IT Support Team",
    "Purchase": "IT Procurement Team",
    "Storage": "Storage and Backup Team",
}


def normalize_text(value):
    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+\.\S+", " email ", text)
    text = re.sub(r"\b\d{7,}\b", " number ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalized_svm_probabilities(scores):
    scores = np.asarray(scores)
    centered = scores - scores.mean(axis=1, keepdims=True)
    scaled = centered / np.maximum(scores.std(axis=1, keepdims=True), 1e-6)
    return softmax(scaled, axis=1)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Advanced category model not found:\n{MODEL_PATH}\n"
        "Run python train_category_model_advanced.py first."
    )

print("Loading advanced category model...")
package = joblib.load(MODEL_PATH)
print("Advanced category model loaded successfully.")


def predict_category(ticket_text):
    text = normalize_text(ticket_text)
    if not text:
        raise ValueError("Ticket text cannot be empty.")

    word = package["word_vectorizer"].transform([text])
    character = package["character_vectorizer"].transform([text])
    exact = hstack([word, character], format="csr")
    similarity = csr_matrix(word @ package["centroids"].T)
    augmented = hstack(
        [exact, similarity * package["centroid_weight"]], format="csr"
    )

    svm_scores = package["svm"].decision_function(augmented)
    svm_probability = normalized_svm_probabilities(svm_scores)

    semantic = package["normalizer"].transform(
        package["svd"].transform(word)
    ).astype(np.float32)
    neural_average = np.mean(
        [model.predict_proba(semantic) for model in package["neural_models"]],
        axis=0,
    )
    neural_probability = softmax(
        np.log(np.maximum(neural_average, 1e-8)) / package["temperature"],
        axis=1,
    )

    scores = (
        (1.0 - package["neural_weight"]) * svm_probability
        + package["neural_weight"] * neural_probability
        + package["class_biases"]
    )
    predicted_index = int(scores.argmax(axis=1)[0])
    ordered = np.sort(scores[0])
    margin = float(ordered[-1] - ordered[-2])

    raw_category = str(package["classes"][predicted_index])
    category = raw_category.replace("internal_it::", "")
    needs_review = margin < float(package["review_threshold"])

    return {
        "category": category,
        "assigned_team": TEAM_MAPPING.get(category, "General IT Support Team"),
        "decision_margin": round(margin, 4),
        "review_threshold": round(float(package["review_threshold"]), 4),
        "needs_review": bool(needs_review),
        "routing_status": (
            "Human review required"
            if needs_review
            else "Automatic routing allowed"
        ),
    }


def main():
    print("\nADVANCED SUPPORT TICKET CLASSIFIER")
    print("=" * 50)
    ticket = input("Enter the complete support ticket:\n> ").strip()
    result = predict_category(ticket)
    print("\n" + "=" * 50)
    print("TICKET ROUTING DECISION")
    print("=" * 50)
    print(f"Category        : {result['category']}")
    print(f"Assigned team   : {result['assigned_team']}")
    print(f"Routing status  : {result['routing_status']}")
    print(f"Decision margin : {result['decision_margin']}")
    print(f"Review threshold: {result['review_threshold']}")


if __name__ == "__main__":
    main()

