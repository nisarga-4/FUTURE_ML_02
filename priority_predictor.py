from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack


PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "priority_v2"
    / "priority_model_v2.joblib"
)


HIGH_URGENCY_WORDS = [
    "urgent", "urgently", "immediate", "immediately",
    "critical", "emergency", "severe", "priority", "asap",
]

OUTAGE_WORDS = [
    "outage", "offline", "system down", "service down",
    "unavailable", "disruption", "failure", "crash",
    "not working",
]

SECURITY_WORDS = [
    "security breach", "breach", "hacked", "malware",
    "ransomware", "unauthorized", "phishing",
    "compromised", "data leak",
]

DATA_LOSS_WORDS = [
    "data loss", "lost data", "deleted data",
    "corrupted", "cannot recover", "backup failed",
]

BLOCKING_WORDS = [
    "blocked", "cannot access", "unable to access",
    "cannot login", "unable to login", "cannot work",
    "stopped working",
]

MULTI_USER_WORDS = [
    "all users", "entire team", "whole team",
    "multiple users", "several users", "company wide",
    "organization wide", "everyone",
]

LOW_URGENCY_WORDS = [
    "general inquiry", "information", "documentation",
    "feature request", "suggestion", "when possible",
    "minor issue", "question about",
]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+\.\S+", " email ", text)
    text = re.sub(r"\b\d{7,}\b", " number ", text)
    text = re.sub(r"[^a-z0-9!?%$€£\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_count(text, keywords):
    return sum(text.count(keyword) for keyword in keywords)


def create_severity_features(subject, body):
    original_text = f"{subject} {body}"
    lower_text = original_text.lower()

    values = [
        min(keyword_count(lower_text, HIGH_URGENCY_WORDS), 5) / 5,
        min(keyword_count(lower_text, OUTAGE_WORDS), 5) / 5,
        min(keyword_count(lower_text, SECURITY_WORDS), 5) / 5,
        min(keyword_count(lower_text, DATA_LOSS_WORDS), 5) / 5,
        min(keyword_count(lower_text, BLOCKING_WORDS), 5) / 5,
        min(keyword_count(lower_text, MULTI_USER_WORDS), 5) / 5,
        min(keyword_count(lower_text, LOW_URGENCY_WORDS), 5) / 5,
        min(original_text.count("!"), 5) / 5,
        min(original_text.count("?"), 5) / 5,
        min(len(original_text.split()), 500) / 500,
    ]

    return csr_matrix(
        np.asarray([values], dtype=np.float32) * 2.0
    )


def calculate_margin(classifier, features):
    if hasattr(classifier, "decision_function"):
        scores = classifier.decision_function(features)
    elif hasattr(classifier, "predict_proba"):
        scores = classifier.predict_proba(features)
    else:
        return 0.0

    scores = np.asarray(scores)

    if scores.ndim == 1:
        return float(abs(scores[0]))

    sorted_scores = np.sort(scores[0])
    return float(sorted_scores[-1] - sorted_scores[-2])


print("Loading Priority V2 model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Priority model not found:\n{MODEL_PATH}"
    )

model_package = joblib.load(MODEL_PATH)

word_vectorizer = model_package["word_vectorizer"]
character_vectorizer = model_package["character_vectorizer"]
type_encoder = model_package["type_encoder"]
classifier = model_package["classifier"]
review_threshold = model_package["review_threshold"]

print("Priority model loaded successfully.")


def predict_priority(subject, body, ticket_type):
    subject = str(subject).strip()
    body = str(body).strip()
    ticket_type = str(ticket_type).strip().title()

    if not subject and not body:
        raise ValueError("Ticket subject or description is required.")

    # Same subject weighting used during training
    model_text = clean_text(
        f"{subject} {subject} {subject} {body}"
    )

    word_features = word_vectorizer.transform([model_text])

    character_features = character_vectorizer.transform(
        [model_text]
    )

    type_frame = pd.DataFrame({
        "type": [ticket_type]
    })

    type_features = type_encoder.transform(type_frame)

    severity_features = create_severity_features(
        subject,
        body,
    )

    final_features = hstack(
        [
            word_features,
            character_features,
            type_features,
            severity_features,
        ],
        format="csr",
    )

    predicted_priority = classifier.predict(
        final_features
    )[0]

    decision_margin = calculate_margin(
        classifier,
        final_features,
    )

    needs_review = decision_margin < review_threshold

    return {
        "priority": str(predicted_priority).title(),
        "ticket_type": ticket_type,
        "decision_margin": round(decision_margin, 4),
        "review_threshold": round(
            float(review_threshold),
            4,
        ),
        "needs_review": bool(needs_review),
        "decision_status": (
            "Human review required"
            if needs_review
            else "Automatic priority assignment allowed"
        ),
    }


def main():
    print("\nSMART PRIORITY PREDICTOR")
    print("=" * 45)

    subject = input("\nTicket subject: ").strip()
    body = input("Ticket description: ").strip()

    print("\nSelect the ticket type:")
    print("1. Incident")
    print("2. Request")
    print("3. Problem")
    print("4. Change")

    choice = input("Enter 1, 2, 3 or 4: ").strip()

    type_mapping = {
        "1": "Incident",
        "2": "Request",
        "3": "Problem",
        "4": "Change",
    }

    ticket_type = type_mapping.get(choice)

    if ticket_type is None:
        print("\nInvalid option. Using 'Incident'.")
        ticket_type = "Incident"

    result = predict_priority(
        subject,
        body,
        ticket_type,
    )

    print("\n" + "=" * 45)
    print("PRIORITY DECISION")
    print("=" * 45)
    print(f"Predicted priority : {result['priority']}")
    print(f"Ticket type       : {result['ticket_type']}")
    print(f"Decision status   : {result['decision_status']}")
    print(f"Decision margin   : {result['decision_margin']}")
    print(f"Review threshold  : {result['review_threshold']}")

    if result["needs_review"]:
        print(
            "\nThe prediction is uncertain, so a support "
            "agent should confirm its priority."
        )
    else:
        print(
            "\nThe prediction is confident enough for "
            "automatic assignment."
        )


if __name__ == "__main__":
    main()