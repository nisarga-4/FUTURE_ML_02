"""SupportFlow AI: description-only Flask inference API and React server."""

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from category_text_predictor import predict_category
from priority_description_predictor import predict_priority


ROOT = Path(__file__).resolve().parent
FRONTEND_DIST = ROOT / "frontend" / "dist"

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIST / "assets"),
    static_url_path="/assets",
)

CORS(app)

RESPONSE_TARGETS = {
    "High": "Respond within 30 minutes",
    "Medium": "Respond within 4 hours",
    "Low": "Respond within 24 hours",
}


def api_error(message, status=400):
    return jsonify({"error": message}), status


def ticket_input_from_request():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("A JSON request body is required.")
    subject = str(data.get("subject", "")).strip()
    description = str(
        data.get("description", data.get("ticket_text", ""))
    ).strip()
    if len(subject) + len(description) < 5:
        raise ValueError("Enter a subject or a ticket description.")
    if len(subject) + len(description) > 5000:
        raise ValueError("Ticket description cannot exceed 5,000 characters.")
    return subject, description


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "application": "SupportFlow AI",
        "input_mode": "description_only",
        "category_model_loaded": True,
        "priority_model_loaded": True,
        "categories": 6,
        "priorities": ["High", "Medium", "Low"],
    })


@app.post("/api/predict")
def predict():
    try:
        subject, description = ticket_input_from_request()
        combined_text = f"{subject}. {description}".strip(". ")
        category = predict_category(subject, description)
        priority = predict_priority(combined_text)
    except (KeyError, TypeError, ValueError) as exc:
        return api_error(str(exc))
    except Exception:
        app.logger.exception("Prediction failed")
        return api_error("Prediction failed. Check the trained model files.", 500)

    category_review = bool(category["needs_review"])
    priority_review = priority["decision_status"] == "Human review required"
    priority_name = priority["priority"]

    return jsonify({
        "prediction": {
            "category": category["category"],
            "assigned_team": category["assigned_team"],
            "priority": priority_name,
            "response_target": RESPONSE_TARGETS[priority_name],
        },
        "category_review": {
            "required": category_review,
            "routing_decision": category["routing_status"],
            "confidence": round(category["confidence"], 6),
            "probabilities": {
                label: round(value, 6)
                for label, value in category["probabilities"].items()
            },
            "threshold": category["review_threshold"],
        },
        "priority_review": {
            "required": priority_review,
            "decision_status": priority["decision_status"],
            "confidence": round(priority["confidence"], 6),
            "threshold": round(priority["review_threshold"], 6),
            "probabilities": {
                label: round(value, 6)
                for label, value in priority["probabilities"].items()
            },
        },
        "extracted_signals": priority["extracted_signals"],
        "human_review_required": category_review or priority_review,
    })


@app.get("/")
def frontend():
    if not (FRONTEND_DIST / "index.html").exists():
        return api_error("React build not found. Run: cd frontend && npm run build", 503)
    return send_from_directory(FRONTEND_DIST, "index.html")


@app.get("/<path:path>")
def frontend_assets(path):
    candidate = FRONTEND_DIST / path
    if candidate.is_file():
        return send_from_directory(FRONTEND_DIST, path)
    if (FRONTEND_DIST / "index.html").exists():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return api_error("React build not found.", 404)


if __name__ == "__main__":
    print("SupportFlow AI description models loaded: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
