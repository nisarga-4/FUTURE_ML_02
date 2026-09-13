# SupportFlow AI

**SupportFlow AI** is an end-to-end machine-learning application that reads a customer-support request and automatically recommends its category, support team, priority, response target, and review status.

The project was developed for **Future Interns â€” Machine Learning Task 2** and is maintained in the repository **`FUTURE_ML_02`**.

## Live demo

- **Application:** [Open SupportFlow AI](https://supportflow-ai-tau.vercel.app)
- **API health check:** [Check the deployed API](https://future-ml-02-nu.vercel.app/health)
- **Repository:** [nisarga-4/FUTURE_ML_02](https://github.com/nisarga-4/FUTURE_ML_02)

> The frontend and prediction API are deployed as separate Vercel projects. The API URL is supplied to the Vite application through `VITE_API_URL`.

## What the system does

Support teams often spend valuable time manually reading, categorizing, prioritizing, and assigning incoming requests. SupportFlow AI turns one natural-language customer statement into an operational decision:

- Predicts one of six customer-support categories
- Assigns the appropriate support team
- Predicts High, Medium, or Low priority
- Recommends a response target
- Displays category and priority confidence
- Flags uncertain decisions for human review
- Recognizes strong qualitative impact language even when exact numbers are missing

## Key features

- Description-first ticket classification
- Six-category customer-support taxonomy
- Hybrid ML and language-guardrail priority inference
- Automatic support-team assignment
- Confidence-based human-review safeguards
- Flask JSON prediction API and health endpoint
- Responsive React and Vite interface
- Purple translucent liquid-glass visual design
- Landing page, name-only entry flow, dashboard, ticket workspace, review queue, analytics, settings, and help pages
- Vercel-hosted frontend and serverless backend

## Supported categories

| Category | Assigned team |
| --- | --- |
| API | API Integration Team |
| Billing | Billing Support Team |
| Cancellation | Retention Team |
| Complaint | Customer Experience Team |
| Technical | Technical Support Team |
| Upgrade | Customer Success Team |

## Priority and response targets

| Priority | Response target | Typical interpretation |
| --- | --- | --- |
| High | 30 minutes | Production outage, widespread impact, security incident, or data loss |
| Medium | 4 hours | A user or team is blocked from completing important work |
| Low | 24 hours | Minor, cosmetic, informational, or non-blocking request |

## Natural-language priority guardrails

The trained priority model performs best when tickets contain operational information such as customers affected, downtime, or error rate. Real users do not always provide those numbers, so the production inference layer includes transparent language guardrails.

Examples:

| Customer statement | Expected priority |
| --- | --- |
| â€œThe production checkout system is down and all customers are blocked.â€ | High |
| â€œMy laptop screen is black, the device will not start, and I cannot work.â€ | Medium |
| â€œThere is a minor spelling mistake on the invoice.â€ | Low |

These guardrails do not replace the trained model. They provide a minimum severity for clear operational phrases and return the reason inside the API's extracted signals.

## Model performance

The following values were obtained from the prepared held-out test datasets:

| Model | Test accuracy | Macro F1 | Selective accuracy | Automated coverage |
| --- | ---: | ---: | ---: | ---: |
| Category model | 99.73% | 99.75% | 99.73% | 100.00% |
| Priority model | 95.29% | 94.71% | 98.18% | 90.92% |

- Category review threshold: approximately `0.4457`
- Priority review threshold: approximately `0.6192`

> Results describe performance on the prepared evaluation datasets. Production performance depends on the quality and similarity of new ticket descriptions. Guardrail confidence values represent deterministic decision strength and should not be interpreted as calibrated model probabilities.

## System workflow

1. The user enters a natural-language customer statement.
2. The React frontend sends the statement to `POST /api/predict`.
3. The Flask API validates and normalizes the request.
4. The category model predicts the category and assigned team.
5. The priority pipeline extracts operational signals and combines structured and text-model scores.
6. Natural-language guardrails handle clear severity phrases when numeric impact details are absent.
7. Review thresholds determine whether automatic routing or human review is appropriate.
8. The interface displays the category, team, priority, response target, and confidence scores.

## Architecture

```text
Browser
   |
   v
React + Vite frontend (Vercel)
   |
   | POST /api/predict
   v
Flask inference API (Vercel)
   |
   +-- Category model -> category + assigned team
   |
   +-- Priority model + language guardrails -> priority + response target
```

## Project structure

```text
FUTURE_ML_02/
â”œâ”€â”€ data/
â”‚   â””â”€â”€ raw/                              # Source datasets
â”œâ”€â”€ frontend/
â”‚   â”œâ”€â”€ public/
â”‚   â”œâ”€â”€ src/
â”‚   â”‚   â”œâ”€â”€ components/
â”‚   â”‚   â”œâ”€â”€ data/
â”‚   â”‚   â”œâ”€â”€ services/
â”‚   â”‚   â”‚   â””â”€â”€ api.js
â”‚   â”‚   â”œâ”€â”€ App.jsx
â”‚   â”‚   â”œâ”€â”€ LandingPage.jsx
â”‚   â”‚   â”œâ”€â”€ landing.css
â”‚   â”‚   â””â”€â”€ styles.css
â”‚   â”œâ”€â”€ package.json
â”‚   â””â”€â”€ vite.config.js
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ category_text/
â”‚   â”‚   â””â”€â”€ category_text_model.joblib
â”‚   â””â”€â”€ priority_description/
â”‚       â””â”€â”€ priority_description_model.joblib
â”œâ”€â”€ category_text_predictor.py
â”œâ”€â”€ priority_description_predictor.py
â”œâ”€â”€ train_category_text_model.py
â”œâ”€â”€ train_priority_description_model.py
â”œâ”€â”€ supportflow_app.py
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ vercel.json
â””â”€â”€ README.md
```

## Technology stack

### Machine learning and backend

- Python
- pandas and NumPy
- scikit-learn
- joblib
- Flask
- Flask-CORS

### Frontend

- React
- Vite
- Framer Motion
- Recharts
- Lucide React
- CSS liquid-glass interface

### Deployment

- GitHub
- Vercel frontend deployment
- Vercel Python serverless API

## Local installation

### 1. Clone the repository

```powershell
git clone https://github.com/nisarga-4/FUTURE_ML_02.git
cd FUTURE_ML_02
```

### 2. Create the Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Start the backend

```powershell
python supportflow_app.py
```

The local API runs at `http://127.0.0.1:5000`.

### 4. Configure and start the frontend

Create `frontend/.env.local`:

```env
VITE_API_URL=http://127.0.0.1:5000
```

Then open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## API reference

### Health check

```http
GET /health
```

Example response:

```json
{
  "application": "SupportFlow AI",
  "category_model_loaded": true,
  "input_mode": "description_only",
  "priority_model_loaded": true,
  "status": "healthy"
}
```

### Predict a ticket

```http
POST /api/predict
Content-Type: application/json
```

Example request:

```json
{
  "subject": "Production payment outage",
  "description": "Our production payment platform is unavailable and all customers are blocked from completing checkout."
}
```

The response includes:

- Predicted category and assigned team
- Predicted priority and response target
- Category and priority confidence values
- Review thresholds and review decisions
- Extracted operational signals
- Priority decision source and guardrail reason when applicable

## Training

The repository contains separate training pipelines for the production models:

```powershell
python train_category_text_model.py
python train_priority_description_model.py
```

Generated artifacts:

```text
models/category_text/category_text_model.joblib
models/priority_description/priority_description_model.joblib
```

The committed model artifacts allow the application to run without retraining. Retraining is required when the datasets, labels, feature extraction, or model configuration changes.

## Vercel configuration

The frontend project uses:

```env
VITE_API_URL=https://future-ml-02-nu.vercel.app
```

The stable production services are:

| Service | URL |
| --- | --- |
| Frontend | `https://supportflow-ai-tau.vercel.app` |
| Backend API | `https://future-ml-02-nu.vercel.app` |

The backend root may report that a React build is unavailable because the frontend is deployed separately. Use `/health` to verify the API.

## Important notes

- The name-only entry screen is a demonstration experience stored in browser local storage; it is not a production authentication system.
- The dashboard and analytics data are demonstration data unless connected to a persistent database.
- Low-confidence model predictions can be held for human review.
- The language guardrails complement the ML model and should be validated regularly against real support-team decisions.
- Never commit `.env` files, access tokens, virtual environments, `node_modules`, or generated build folders.

## Future improvements

- Persist tickets and review decisions in a database
- Add authenticated user and team accounts
- Capture human corrections for model monitoring and retraining
- Add drift detection and production performance monitoring
- Expand automated tests for API and guardrail behaviour
- Add rate limiting and structured application logging

## License

See [LICENSE](LICENSE).