"""Infer priority and operational signals from one ticket description."""

from datetime import datetime
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "priority_description" / "priority_description_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Description-priority model not found: {MODEL_PATH}\n"
        "Run: python train_priority_description_model.py"
    )

print("Loading description-priority model...")
package = joblib.load(MODEL_PATH)
print("Description-priority model loaded successfully.")


def _number(pattern, text, default):
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1).replace(",", "")) if match else default


def _contains(text, words):
    return any(word in text for word in words)


def _category(name, value):
    mapping = package["category_maps"][name]
    key = str(value).strip().lower()
    if key not in mapping:
        key = next(iter(mapping))
    return mapping[key]


def _bucket(value, limits, names):
    return names[int(np.searchsorted(limits, float(value), side="right"))]


def extract_signals(ticket_text):
    text = re.sub(r"\s+", " ", str(ticket_text)).strip()
    lower = text.lower()

    affected_match = re.search(
        r"(?:approximately|about|around|over|more than|nearly)?\s*"
        r"(\d[\d,]*)\s+(?:customers?|users?|accounts?|employees?|workers?|"
        r"orders?|requests?|transactions?|deliveries?|shipments?|devices?|seats?)",
        lower,
    )
    affected = int(affected_match.group(1).replace(",", "")) if affected_match else 1

    error_rate = _number(
        r"(\d+(?:\.\d+)?)\s*(?:%|percent)(?:\s+(?:error|failure)\s*rate)?",
        lower, 0.0,
    )
    downtime_match = re.search(
        r"(?:unavailable|offline|down|downtime|outage)?(?:\s+for)?\s*"
        r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?|days?)",
        lower,
    )
    downtime = 0.0
    if downtime_match:
        downtime = float(downtime_match.group(1))
        unit = downtime_match.group(2)
        if unit.startswith(("hour", "hr")):
            downtime *= 60
        elif unit.startswith("day"):
            downtime *= 1440

    payment = _contains(lower, [
        "payment", "checkout", "transaction", "billing", "invoice",
        "refund", "purchase", "revenue",
    ])
    security = _contains(lower, [
        "security", "breach", "phishing", "malware", "ransomware",
        "unauthorized", "unauthorised", "compromised",
    ])
    data_loss = _contains(lower, [
        "data loss", "lost data", "deleted data", "missing data",
        "corrupted", "corruption",
    ])
    disrupted = _contains(lower, [
        "production", "unavailable", "offline", "outage", "down",
        "cannot", "can't", "failing", "failure", "failed", "blocked",
        "stopped", "rejected", "disconnect", "not working", "not resolved",
    ])

    if _contains(lower, ["api", "webhook", "endpoint", "http 5", "rate limit"]):
        product_area = "api"
    elif _contains(lower, ["billing", "payment", "invoice", "checkout", "transaction"]):
        product_area = "billing"
    elif _contains(lower, ["login", "password", "authentication", "account", "access"]):
        product_area = "auth"
    elif _contains(lower, ["notification", "email alert", "sms", "push message"]):
        product_area = "notifications"
    elif _contains(lower, ["pipeline", "etl", "warehouse", "data job"]):
        product_area = "data_pipeline"
    elif _contains(lower, ["mobile", "android", "ios", "app"]):
        product_area = "mobile"
    else:
        product_area = "analytics"

    enterprise = _contains(lower, ["enterprise", "vip", "key account", "executive", "c-level"])
    company_size = "Large" if enterprise else "Medium"
    customer_tier = "Enterprise" if enterprise else "Plus"
    industry = (
        "fintech" if _contains(lower, ["fintech", "bank", "financial"])
        else "ecommerce" if _contains(lower, ["ecommerce", "checkout", "cart", "store"])
        else "healthcare" if _contains(lower, ["healthcare", "hospital", "patient"])
        else "logistics" if _contains(lower, ["logistics", "shipment", "delivery"])
        else "gaming" if _contains(lower, ["gaming", "game", "player"])
        else "media" if _contains(lower, ["media", "stream", "video"])
        else "saas_b2b"
    )
    region = "EMEA" if "emea" in lower else "AMER" if _contains(lower, ["amer", "america", "usa", "canada"]) else "APAC"
    channel = "email" if "email" in lower else "chat" if "chat" in lower else "phone" if _contains(lower, ["phone", "call"]) else "web"
    role = (
        "c_level" if _contains(lower, ["c-level", "ceo", "cto", "executive"])
        else "devops" if _contains(lower, ["devops", "sre", "engineer"])
        else "finance" if _contains(lower, ["finance team", "accounting"])
        else "product_manager" if _contains(lower, ["product manager", "product owner"])
        else "support"
    )
    sentiment = "negative" if disrupted or security or error_rate >= 25 else "neutral"
    runbook = _contains(lower, ["runbook", "playbook", "documented procedure"])

    return {
        "raw_text": text,
        "day_of_week_num": datetime.now().isoweekday(),
        "company_size": company_size,
        "industry": industry,
        "customer_tier": customer_tier,
        "org_users": max(100, affected),
        "region": region,
        "past_30d_tickets": int(_number(r"(\d+)\s+(?:tickets?|issues?)\s+(?:in|during)\s+(?:the\s+)?(?:past|last)\s+30\s+days", lower, 0)),
        "past_90d_incidents": 1 if disrupted else 0,
        "product_area": product_area,
        "booking_channel": channel,
        "reported_by_role": role,
        "customers_affected": affected,
        "error_rate_pct": min(100.0, max(0.0, error_rate)),
        "downtime_min": max(0.0, downtime),
        "payment_impact_flag": int(payment),
        "security_incident_flag": int(security),
        "data_loss_flag": int(data_loss),
        "has_runbook": int(runbook),
        "customer_sentiment": sentiment,
        "description_length": len(text),
        "stated_customers_affected": affected if affected_match else None,
        "stated_error_rate_pct": error_rate if re.search(r"\d+(?:\.\d+)?\s*(?:%|percent)", lower) else None,
        "stated_downtime_min": downtime if downtime_match else None,
    }


def _canonical(signals):
    impact = _bucket(signals["customers_affected"], [10, 100, 1000], ["tiny", "small", "large", "massive"])
    error = _bucket(signals["error_rate_pct"], [5, 25, 60], ["low", "moderate", "high", "critical"])
    downtime = _bucket(signals["downtime_min"], [1, 30, 120], ["none", "short", "long", "critical"])
    return (
        f"{signals['raw_text']} {signals['product_area']} support incident affecting "
        f"{signals['customers_affected']} customers. Error rate {signals['error_rate_pct']} percent. "
        f"Service unavailable for {signals['downtime_min']} minutes. Customer tier "
        f"{signals['customer_tier']}; {signals['company_size']} {signals['industry']} company "
        f"in {signals['region']}. Payment impact {signals['payment_impact_flag']}; security incident "
        f"{signals['security_incident_flag']}; data loss {signals['data_loss_flag']}; runbook "
        f"{signals['has_runbook']}; sentiment {signals['customer_sentiment']}. "
        f"IMPACT_{impact} ERROR_{error} DOWNTIME_{downtime}"
    )


def infer_priority_guardrail(signals):
    """Infer minimum priority from strong natural-language impact signals."""

    text = signals["raw_text"].lower()

    widespread_phrases = (
        "company-wide",
        "company wide",
        "system-wide",
        "system wide",
        "all users",
        "all customers",
        "all employees",
        "everyone is blocked",
        "entire organization",
        "complete outage",
    )

    blocked_work_phrases = (
        "black screen",
        "will not start",
        "won't start",
        "cannot work",
        "can't work",
        "unable to work",
        "blocked from working",
        "completely unusable",
        "cannot access",
        "can't access",
        "unable to access",
    )

    low_impact_phrases = (
        "minor issue",
        "cosmetic issue",
        "spelling mistake",
        "general question",
        "feature request",
        "no users are blocked",
        "no downtime",
    )

    production_outage = (
        "production" in text
        and any(
            phrase in text
            for phrase in (
                "down",
                "unavailable",
                "outage",
                "not working",
                "failing",
                "failed",
            )
        )
    )

    if signals["security_incident_flag"] or signals["data_loss_flag"]:
        return "High", "Security or data-loss incident"

    if production_outage or any(phrase in text for phrase in widespread_phrases):
        return "High", "Production or widespread service disruption"

    if any(phrase in text for phrase in blocked_work_phrases):
        return "Medium", "User is blocked from performing required work"

    if any(phrase in text for phrase in low_impact_phrases):
        return "Low", "Minor issue without operational disruption"

    return None, None


def predict_priority(ticket_text):
    signals = extract_signals(ticket_text)
    row = {
        "day_of_week_num": signals["day_of_week_num"],
        "company_size_cat": _category("company_size", signals["company_size"]),
        "industry_cat": _category("industry", signals["industry"]),
        "customer_tier_cat": _category("customer_tier", signals["customer_tier"]),
        "org_users": signals["org_users"],
        "region_cat": _category("region", signals["region"]),
        "past_30d_tickets": signals["past_30d_tickets"],
        "past_90d_incidents": signals["past_90d_incidents"],
        "product_area_cat": _category("product_area", signals["product_area"]),
        "booking_channel_cat": _category("booking_channel", signals["booking_channel"]),
        "reported_by_role_cat": _category("reported_by_role", signals["reported_by_role"]),
        "customers_affected": signals["customers_affected"],
        "error_rate_pct": signals["error_rate_pct"],
        "downtime_min": signals["downtime_min"],
        "payment_impact_flag": signals["payment_impact_flag"],
        "security_incident_flag": signals["security_incident_flag"],
        "data_loss_flag": signals["data_loss_flag"],
        "has_runbook": signals["has_runbook"],
        "customer_sentiment_cat": _category("customer_sentiment", signals["customer_sentiment"]),
        "description_length": signals["description_length"],
    }
    frame = pd.DataFrame([row], columns=package["features"])
    structured = package["structured_model"].predict_proba(frame)[0]
    text_features = package["text_vectorizer"].transform([_canonical(signals)])
    text_probability = package["text_model"].predict_proba(text_features)[0]
    weight = float(package["text_weight"])
    probabilities = (1.0 - weight) * structured + weight * text_probability

    classes = package["classes"]
    class_names = [str(label).title() for label in classes]
    best = int(probabilities.argmax())

    guardrail_priority, guardrail_reason = infer_priority_guardrail(signals)

    if guardrail_priority in class_names:
        target_index = class_names.index(guardrail_priority)
        confidence_floor = {
            "High": 0.85,
            "Medium": 0.75,
            "Low": 0.70,
        }[guardrail_priority]

        adjusted = probabilities.copy()
        target_confidence = max(
            float(adjusted[target_index]),
            confidence_floor,
        )
        other_total = float(adjusted.sum() - adjusted[target_index])
        remaining = 1.0 - target_confidence

        if other_total > 0:
            for index in range(len(adjusted)):
                if index != target_index:
                    adjusted[index] = (
                        adjusted[index] / other_total
                    ) * remaining
        else:
            for index in range(len(adjusted)):
                if index != target_index:
                    adjusted[index] = remaining / (len(adjusted) - 1)

        adjusted[target_index] = target_confidence
        probabilities = adjusted
        best = target_index

    confidence = float(probabilities[best])
    threshold = float(package["review_threshold"])
    public_signals = {
        "customers_affected": signals["stated_customers_affected"],
        "error_rate_pct": signals["stated_error_rate_pct"],
        "downtime_min": signals["stated_downtime_min"],
        "product_area": signals["product_area"],
        "payment_impact": bool(signals["payment_impact_flag"]),
        "security_incident": bool(signals["security_incident_flag"]),
        "data_loss": bool(signals["data_loss_flag"]),
        "priority_source": (
            "language_guardrail" if guardrail_priority else "trained_model"
        ),
        "priority_reason": guardrail_reason,
    }
    return {
        "priority": str(classes[best]).title(),
        "confidence": confidence,
        "review_threshold": threshold,
        "decision_status": "Automatic priority assignment allowed" if confidence >= threshold else "Human review required",
        "probabilities": {str(label).title(): float(value) for label, value in zip(classes, probabilities)},
        "extracted_signals": public_signals,
    }
