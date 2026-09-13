from pathlib import Path
import json

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from ticket_system import (
    TEAM_MAPPING,
    calculate_priority,
    transform_ticket,
)

ROOT = Path(__file__).resolve().parent

# Use the evaluated model because the review threshold was
# measured specifically for this model.
MODEL_PATH = (
    ROOT
    / "models"
    / "semantic_clusters"
    / "best_semantic_cluster_model.joblib"
)

THRESHOLD_PATH = (
    ROOT / "models" / "review_threshold.json"
)

app = Flask(__name__)

print("Loading ticket classification model...")
model_bundle = joblib.load(MODEL_PATH)

with open(
    THRESHOLD_PATH,
    "r",
    encoding="utf-8",
) as file:
    threshold_information = json.load(file)

REVIEW_THRESHOLD = threshold_information[
    "decision_margin_threshold"
]

print("Model loaded successfully.")
print(f"Review threshold: {REVIEW_THRESHOLD:.4f}")


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "categories": 8,
        "review_threshold": REVIEW_THRESHOLD,
    })


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "A JSON request body is required."
        }), 400

    ticket_text = str(
        data.get("ticket_text", "")
    ).strip()

    impact = str(
        data.get("impact", "individual")
    ).lower().strip()

    urgency = str(
        data.get("urgency", "normal")
    ).lower().strip()

    if len(ticket_text) < 5:
        return jsonify({
            "error": (
                "Ticket description must contain "
                "at least five characters."
            )
        }), 400

    valid_impacts = {
        "individual",
        "team",
        "company",
    }

    valid_urgencies = {
        "normal",
        "soon",
        "blocked",
    }

    if impact not in valid_impacts:
        return jsonify({
            "error": (
                "Impact must be individual, team, or company."
            )
        }), 400

    if urgency not in valid_urgencies:
        return jsonify({
            "error": (
                "Urgency must be normal, soon, or blocked."
            )
        }), 400

    features = transform_ticket(
        model_bundle,
        ticket_text,
    )

    classifier = model_bundle["classifier"]

    category_id = classifier.predict(features)[0]
    category = category_id.replace("internal_it::", "")

    decision_scores = classifier.decision_function(
        features
    )[0]

    sorted_scores = np.sort(decision_scores)[::-1]

    decision_margin = float(
        sorted_scores[0] - sorted_scores[1]
    )

    if decision_margin >= REVIEW_THRESHOLD:
        routing_decision = "Automatic"
        review_required = False
    else:
        routing_decision = "Manual review"
        review_required = True

    priority, response_target, reasons = (
        calculate_priority(
            ticket_text,
            impact,
            urgency,
        )
    )

    return jsonify({
        "ticket": {
            "description": ticket_text,
            "impact": impact,
            "urgency": urgency,
        },
        "prediction": {
            "category": category,
            "assigned_team": TEAM_MAPPING.get(
                category,
                "General Support Team",
            ),
            "priority": priority,
            "response_target": response_target,
        },
        "review": {
            "required": review_required,
            "routing_decision": routing_decision,
            "decision_margin": round(
                decision_margin,
                4,
            ),
            "threshold": round(
                REVIEW_THRESHOLD,
                4,
            ),
        },
        "priority_reasons": reasons,
    })


@app.errorhandler(404)
def page_not_found(error):
    return jsonify({
        "error": "The requested page was not found."
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "An internal server error occurred."
    }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )