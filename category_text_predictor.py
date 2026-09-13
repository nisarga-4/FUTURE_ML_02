"""Production inference helper for the calibrated category text model."""

from pathlib import Path
import re

import joblib


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "category_text" / "category_text_model.joblib"

TEAM_MAPPING = {
    "technical": "Technical Support Team",
    "billing": "Billing Support Team",
    "api": "API Integration Team",
    "upgrade": "Customer Success Team",
    "complaint": "Customer Experience Team",
    "cancellation": "Retention Team",
}

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Category text model not found: {MODEL_PATH}\n"
        "Run: python train_category_text_model.py"
    )

package = joblib.load(MODEL_PATH)


def normalize(value):
    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+\.\S+", " email ", text)
    text = re.sub(r"\b\d{7,}\b", " number ", text)
    return re.sub(r"\s+", " ", text).strip()


def predict_category(subject, description):
    combined = normalize(f"{subject}. {description}")
    if len(combined) < 5:
        raise ValueError("Subject and description cannot both be empty.")
    features = package["vectorizer"].transform([combined])
    probabilities = package["model"].predict_proba(features)[0]
    classes = package["classes"]
    best = int(probabilities.argmax())
    category = str(classes[best])
    confidence = float(probabilities[best])
    threshold = float(package["review_threshold"])
    return {
        "category": category.title(),
        "assigned_team": TEAM_MAPPING.get(category, "General Support Team"),
        "confidence": confidence,
        "probabilities": {
            str(label).title(): float(value)
            for label, value in zip(classes, probabilities)
        },
        "needs_review": confidence < threshold,
        "review_threshold": threshold,
        "routing_status": (
            "Human review required"
            if confidence < threshold
            else "Automatic routing allowed"
        ),
    }
