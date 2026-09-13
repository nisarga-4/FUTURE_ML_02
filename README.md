# SupportFlow AI

Premium purple liquid-glass React + Vite frontend for AI support-ticket classification and prioritization.

## Run

```bash
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally http://localhost:5173).

## Build

```bash
npm run build
npm run preview
```

## Backend integration

Replace the mock implementation in:

`src/services/api.js`

with your Flask/FastAPI endpoint. For example, use `VITE_API_URL` in a `.env` file.

## Main structure

- `src/components` — reusable glass UI components
- `src/pages` — Dashboard, New Ticket, Review Queue, Analytics, Settings, Help
- `src/services/api.js` — clean backend integration point
- `src/theme.js` — centralized design values
- `src/styles.css` — liquid-glass system and responsive styling
- `src/data/mockData.js` — realistic mock ticket and analytics data

## Tech

React, Vite, Tailwind CSS, Lucide React, Recharts, Framer Motion, React Router.
