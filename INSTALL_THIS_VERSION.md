# SupportFlow AI — Liquid Glass React Upgrade

Copy every file and folder from this extracted package into:

```text
D:\support_ticket_project
```

This package does not include or replace `models`, `data`, or `reports`.

## Run with React + Vite

Open two VS Code terminals.

Terminal 1 — ML backend:

```powershell
cd D:\support_ticket_project
.venv\Scripts\activate
pip install -r requirements_supportflow.txt
python supportflow_app.py
```

Terminal 2 — React frontend:

```powershell
cd D:\support_ticket_project\frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Run the compiled production version

The production build is already included. For a single-terminal run:

```powershell
cd D:\support_ticket_project
.venv\Scripts\activate
python supportflow_app.py
```

Open `http://127.0.0.1:5000`.

