# SupportFlow AI — Pearl Pink Liquid Glass

This is the final React + Vite visual upgrade. It does not include or replace
your `models`, `data`, or `reports` folders.

## Install

1. Stop the old servers with `Ctrl+C`.
2. Extract this ZIP.
3. Copy everything inside the extracted folder.
4. Paste into `D:\support_ticket_project` and replace matching files.

## Run

Terminal 1:

```powershell
cd D:\support_ticket_project
.venv\Scripts\activate
pip install -r requirements_supportflow.txt
python supportflow_app.py
```

Terminal 2:

```powershell
cd D:\support_ticket_project\frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

The production build is included and can also be viewed from
`http://127.0.0.1:5000` while `supportflow_app.py` is running.

