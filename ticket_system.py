from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "production_ticket_model.joblib"

TEAM_MAPPING = {
    "Access": "Identity and Access Team",
    "Administrative rights": "Privileged Access Team",
    "HR Support": "HR Service Desk",
    "Hardware": "Hardware Support Team",
    "Internal Project": "Internal Projects Team",
    "Miscellaneous": "General Support Team",
    "Purchase": "Procurement Team",
    "Storage": "Storage and Backup Team",
}

CRITICAL_TERMS = {
    "security breach",
    "ransomware",
    "data loss",
    "data deleted",
    "system outage",
    "server down",
    "entire company",
    "all employees",
    "multiple users",
    "cannot work",
    "completely blocked",
    "urgent",
    "emergency",
}

FAILURE_TERMS = {
    "not working",
    "unable to access",
    "cannot access",
    "login failed",
    "activation failed",
    "system failed",
    "error",
    "crashed",
    "offline",
    "locked",
}

ROUTINE_TERMS = {
    "information",
    "general enquiry",
    "new purchase",
    "quotation",
    "when possible",
    "routine request",
}


def transform_ticket(bundle, ticket_text):
    texts = [ticket_text]

    exact = bundle["exact_features"].transform(texts)

    word_features = bundle[
        "semantic_vectorizer"
    ].transform(texts)

    lsa = bundle["svd"].transform(word_features)
    lsa = bundle["normalizer"].transform(lsa)
    lsa = lsa.astype(np.float32)

    similarities = lsa @ bundle["centroids"].T

    features = hstack([
        exact,
        csr_matrix(lsa * 2.0),
        csr_matrix(
            similarities.astype(np.float32) * 2.0
        ),
    ]).tocsr()

    return features


def calculate_priority(ticket_text, impact, urgency):
    """
    Transparent operational priority rules.

    Impact:
        individual, team, company

    Urgency:
        normal, soon, blocked
    """

    text = ticket_text.lower()

    impact_scores = {
        "individual": 0,
        "team": 2,
        "company": 4,
    }

    urgency_scores = {
        "normal": 0,
        "soon": 1,
        "blocked": 3,
    }

    score = impact_scores[impact] + urgency_scores[urgency]
    reasons = [
        f"Impact: {impact}",
        f"Urgency: {urgency}",
    ]

    critical_matches = [
        term for term in CRITICAL_TERMS if term in text
    ]

    failure_matches = [
        term for term in FAILURE_TERMS if term in text
    ]

    routine_matches = [
        term for term in ROUTINE_TERMS if term in text
    ]

    if critical_matches:
        score += 4
        reasons.append(
            "Critical wording: " + ", ".join(critical_matches)
        )

    elif failure_matches:
        score += 2
        reasons.append(
            "Service failure: " + ", ".join(failure_matches)
        )

    if routine_matches and urgency == "normal":
        score -= 1
        reasons.append("Routine request wording detected")

    if score >= 5:
        priority = "High"
        response_target = "Respond within 30 minutes"

    elif score >= 2:
        priority = "Medium"
        response_target = "Respond within 4 hours"

    else:
        priority = "Low"
        response_target = "Respond within 24 hours"

    return priority, response_target, reasons


def predict_ticket(bundle, ticket_text, impact, urgency):
    features = transform_ticket(bundle, ticket_text)
    classifier = bundle["classifier"]

    category_id = classifier.predict(features)[0]
    category = category_id.replace("internal_it::", "")

    # Linear SVM scores are decision margins, not probabilities.
    decision_scores = classifier.decision_function(features)[0]
    ordered_scores = np.sort(decision_scores)[::-1]

    decision_margin = float(
        ordered_scores[0] - ordered_scores[1]
    )

    if decision_margin < 0.25:
        review_status = "Manual review recommended"
    else:
        review_status = "Automatic routing allowed"

    priority, response_target, reasons = calculate_priority(
        ticket_text,
        impact,
        urgency,
    )

    return {
        "category": category,
        "team": TEAM_MAPPING.get(
            category,
            "General Support Team",
        ),
        "priority": priority,
        "response_target": response_target,
        "review_status": review_status,
        "decision_margin": decision_margin,
        "priority_reasons": reasons,
    }


def select_option(title, options):
    print(f"\n{title}")

    for number, option in enumerate(options, start=1):
        print(f"{number}. {option.title()}")

    while True:
        choice = input("Select an option: ").strip()

        if choice.isdigit():
            index = int(choice) - 1

            if 0 <= index < len(options):
                return options[index]

        print("Please enter a valid option number.")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found:\n{MODEL_PATH}"
        )

    print("Loading production ticket model...")
    bundle = joblib.load(MODEL_PATH)

    print("\nSMART SUPPORT TICKET SYSTEM")
    print("=" * 45)

    while True:
        ticket_text = input(
            "\nEnter the support ticket:\n> "
        ).strip()

        if len(ticket_text) < 5:
            print("Please enter a meaningful ticket description.")
            continue

        impact = select_option(
            "Who is affected?",
            ["individual", "team", "company"],
        )

        urgency = select_option(
            "How urgent is the issue?",
            ["normal", "soon", "blocked"],
        )

        result = predict_ticket(
            bundle,
            ticket_text,
            impact,
            urgency,
        )

        print("\n" + "=" * 45)
        print("TICKET DECISION")
        print("=" * 45)
        print(f"Category       : {result['category']}")
        print(f"Assigned team  : {result['team']}")
        print(f"Priority       : {result['priority']}")
        print(f"Response target: {result['response_target']}")
        print(f"Review status  : {result['review_status']}")
        print(
            f"Decision margin: "
            f"{result['decision_margin']:.3f}"
        )

        print("\nPriority reasons:")
        for reason in result["priority_reasons"]:
            print(f"- {reason}")

        again = input(
            "\nClassify another ticket? (yes/no): "
        ).strip().lower()

        if again not in {"yes", "y"}:
            print("\nTicket system closed.")
            break


if __name__ == "__main__":
    main()