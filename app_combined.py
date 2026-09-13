"""Flask API combining advanced category and business-priority models."""

from flask import Flask, jsonify, render_template, request

from category_predictor_advanced import predict_category
from priority_business_predictor import predict_priority

BUSINESS_FIELDS = {
    "company_size",
    "industry",
    "customer_tier",
    "org_users",
    "region",
    "past_30d_tickets",
    "past_90d_incidents",
    "product_area",
    "booking_channel",
    "reported_by_role",
    "customers_affected",
    "error_rate_pct",
    "downtime_min",
    "payment_impact",
    "security_incident",
    "data_loss",
    "has_runbook",
    "customer_sentiment",
}

app = Flask(__name__)
print("Advanced category and business-priority models loaded successfully.")


def error_response(message, status=400):
    return jsonify({"error": message}), status


def validate_category_input(data):
    ticket_text = str(data.get("ticket_text", "")).strip()
    if len(ticket_text) < 5:
        raise ValueError("Ticket description must contain at least five characters.")
    return ticket_text


def priority_prediction(data, ticket_text):
    missing = sorted(BUSINESS_FIELDS - set(data))
    if missing:
        raise ValueError(
            "Missing business-priority fields: " + ", ".join(missing)
        )
    ticket = {field: data[field] for field in BUSINESS_FIELDS}
    ticket["description"] = ticket_text
    result = predict_priority(ticket)
    priority = result["priority"]
    response_targets = {
        "High": "Respond within 30 minutes",
        "Medium": "Respond within 4 hours",
        "Low": "Respond within 24 hours",
    }
    return {
        "priority": priority,
        "response_target": response_targets[priority],
        "confidence": round(result["confidence"], 4),
        "probabilities": {
            label: round(value, 4)
            for label, value in result["probabilities"].items()
        },
        "review_threshold": round(result["review_threshold"], 4),
        "review_required": result["decision_status"] == "Human review required",
        "decision_status": result["decision_status"],
    }


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "category_model_loaded": True,
        "priority_model_loaded": True,
        "categories": 8,
        "priorities": ["High", "Medium", "Low"],
        "category_model": "advanced SVM + semantic neural blend",
        "priority_model": "business-impact gradient boosting",
    })


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error_response("A JSON request body is required.")
    try:
        ticket_text = validate_category_input(data)
        category = predict_category(ticket_text)
        priority = priority_prediction(data, ticket_text)
    except (KeyError, TypeError, ValueError) as error:
        return error_response(str(error))
    except Exception:
        app.logger.exception("Prediction failed")
        return error_response("Prediction failed. Check the supplied field values.", 500)

    overall_review = category["needs_review"] or priority["review_required"]
    return jsonify({
        "ticket": {"description": ticket_text},
        "prediction": {
            "category": category["category"],
            "assigned_team": category["assigned_team"],
            "priority": priority["priority"],
            "response_target": priority["response_target"],
        },
        "category_review": {
            "required": category["needs_review"],
            "routing_decision": category["routing_status"],
            "decision_margin": category["decision_margin"],
            "threshold": category["review_threshold"],
        },
        "priority_review": {
            "required": priority["review_required"],
            "decision_status": priority["decision_status"],
            "confidence": priority["confidence"],
            "threshold": priority["review_threshold"],
            "probabilities": priority["probabilities"],
        },
        "human_review_required": overall_review,
    })


@app.errorhandler(404)
def page_not_found(_error):
    return error_response("The requested page was not found.", 404)


@app.errorhandler(500)
def internal_error(_error):
    return error_response("An internal server error occurred.", 500)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
