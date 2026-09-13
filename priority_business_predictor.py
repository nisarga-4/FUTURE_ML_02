"""Inference helper for the structured business-impact priority model."""

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "priority_business" / "priority_business_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Business priority model not found:\n{MODEL_PATH}\n"
        "Run python train_priority_business_model.py first."
    )

print("Loading business-priority model...")
package = joblib.load(MODEL_PATH)
print("Business-priority model loaded successfully.")


def _category(name, value):
    mapping = package["category_maps"][name]
    key = str(value).strip().lower()
    if key not in mapping:
        raise ValueError(f"Invalid {name}: {value!r}. Choose from {sorted(mapping)}")
    return mapping[key]


def _flag(value):
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "yes", "y", "true"}:
        return 1
    if text in {"0", "no", "n", "false"}:
        return 0
    raise ValueError(f"Expected yes/no value, received {value!r}")


def predict_priority(ticket):
    row = {
        "day_of_week_num": int(ticket.get("day_of_week_num", datetime.now().isoweekday())),
        "company_size_cat": _category("company_size", ticket["company_size"]),
        "industry_cat": _category("industry", ticket["industry"]),
        "customer_tier_cat": _category("customer_tier", ticket["customer_tier"]),
        "org_users": int(ticket["org_users"]),
        "region_cat": _category("region", ticket["region"]),
        "past_30d_tickets": int(ticket["past_30d_tickets"]),
        "past_90d_incidents": int(ticket["past_90d_incidents"]),
        "product_area_cat": _category("product_area", ticket["product_area"]),
        "booking_channel_cat": _category("booking_channel", ticket["booking_channel"]),
        "reported_by_role_cat": _category("reported_by_role", ticket["reported_by_role"]),
        "customers_affected": int(ticket["customers_affected"]),
        "error_rate_pct": float(ticket["error_rate_pct"]),
        "downtime_min": float(ticket["downtime_min"]),
        "payment_impact_flag": _flag(ticket["payment_impact"]),
        "security_incident_flag": _flag(ticket["security_incident"]),
        "data_loss_flag": _flag(ticket["data_loss"]),
        "has_runbook": _flag(ticket["has_runbook"]),
        "customer_sentiment_cat": _category("customer_sentiment", ticket.get("customer_sentiment", "unknown")),
        "description_length": len(str(ticket.get("description", "")).strip()),
    }
    frame = pd.DataFrame([row], columns=package["features"])
    probabilities = package["model"].predict_proba(frame)[0]
    classes = package["classes"]
    best = int(probabilities.argmax())
    confidence = float(probabilities[best])
    threshold = float(package["review_threshold"])
    return {
        "priority": str(classes[best]).title(),
        "confidence": confidence,
        "decision_status": "Automatic priority assignment allowed" if confidence >= threshold else "Human review required",
        "review_threshold": threshold,
        "probabilities": {str(label).title(): float(probability) for label, probability in zip(classes, probabilities)},
    }
