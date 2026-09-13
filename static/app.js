const form = document.getElementById("ticket-form");
const button = document.getElementById("analyse-button");
const buttonLabel = button.querySelector("span");
const errorMessage = document.getElementById("error-message");

const emptyState = document.getElementById("empty-state");
const resultPanel = document.getElementById("result-panel");

const value = (id) => {
    return document.getElementById(id).value;
};

const numberValue = (id) => {
    return Number(value(id));
};

const checked = (id) => {
    return document.getElementById(id).checked;
};

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    errorMessage.textContent = "";
    button.disabled = true;
    buttonLabel.textContent = "Analysing ticket...";

    const payload = {
        ticket_text: value("ticket-text").trim(),

        company_size: value("company-size"),
        industry: value("industry"),
        customer_tier: value("customer-tier"),
        org_users: numberValue("org-users"),
        region: value("region"),

        past_30d_tickets:
            numberValue("past-30d-tickets"),

        past_90d_incidents:
            numberValue("past-90d-incidents"),

        product_area: value("product-area"),
        booking_channel: value("booking-channel"),
        reported_by_role: value("reported-by-role"),

        customers_affected:
            numberValue("customers-affected"),

        error_rate_pct:
            numberValue("error-rate-pct"),

        downtime_min:
            numberValue("downtime-min"),

        payment_impact:
            checked("payment-impact"),

        security_incident:
            checked("security-incident"),

        data_loss:
            checked("data-loss"),

        has_runbook:
            checked("has-runbook"),

        customer_sentiment:
            value("customer-sentiment")
    };

    try {
        const response = await fetch("/api/predict", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Prediction failed."
            );
        }

        showResult(data);
    } catch (error) {
        errorMessage.textContent = error.message;
    } finally {
        button.disabled = false;
        buttonLabel.textContent =
            "Analyse and route ticket";
    }
});

function showResult(data) {
    emptyState.classList.add("hidden");
    resultPanel.classList.remove("hidden");

    document.getElementById(
        "result-category"
    ).textContent = data.prediction.category;

    document.getElementById(
        "result-team"
    ).textContent = data.prediction.assigned_team;

    const priorityElement = document.getElementById(
        "result-priority"
    );

    priorityElement.textContent =
        data.prediction.priority;

    priorityElement.dataset.priority =
        data.prediction.priority.toLowerCase();

    document.getElementById(
        "result-response"
    ).textContent = data.prediction.response_target;

    document.getElementById(
        "result-margin"
    ).textContent =
        data.category_review.decision_margin.toFixed(4);

    document.getElementById(
        "result-confidence"
    ).textContent = percent(
        data.priority_review.confidence
    );

    const reviewRequired =
        data.human_review_required;

    const badge = document.getElementById(
        "review-badge"
    );

    if (reviewRequired) {
        badge.textContent = "Review required";
        badge.className = "badge review";
    } else {
        badge.textContent = "Automatic decision";
        badge.className = "badge automatic";
    }

    setReviewMessage(
        "category-review-message",

        data.category_review.required,

        "Category: human confirmation required.",

        "Category: automatic routing allowed. " +
        "Decision margin: " +
        data.category_review.decision_margin.toFixed(4)
    );

    setReviewMessage(
        "priority-review-message",

        data.priority_review.required,

        "Priority: human confirmation required.",

        "Priority: automatic assignment allowed. " +
        "Confidence: " +
        percent(data.priority_review.confidence)
    );

    showProbabilities(
        data.priority_review.probabilities
    );

    resultPanel.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}

function showProbabilities(probabilities) {
    const probabilityList = document.getElementById(
        "probability-list"
    );

    probabilityList.innerHTML = "";

    Object.entries(probabilities).forEach(
        ([label, probability]) => {
            const row = document.createElement("div");

            row.className = "probability-row";

            const name = document.createElement("span");
            name.textContent = label;

            const track = document.createElement("div");
            const fill = document.createElement("i");

            fill.style.width =
                `${probability * 100}%`;

            track.appendChild(fill);

            const score = document.createElement("strong");
            score.textContent = percent(probability);

            row.appendChild(name);
            row.appendChild(track);
            row.appendChild(score);

            probabilityList.appendChild(row);
        }
    );
}

function setReviewMessage(
    id,
    warning,
    warningText,
    successText
) {
    const element = document.getElementById(id);

    element.textContent = warning
        ? warningText
        : successText;

    element.className = warning
        ? "review-message warning"
        : "review-message";
}

function percent(number) {
    return `${(Number(number) * 100).toFixed(2)}%`;
}