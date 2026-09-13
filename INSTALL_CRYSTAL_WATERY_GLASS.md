# SupportFlow AI — Premium Blush Crystal Glass

This is the polished light-pink, maximum-transparency React + Vite edition.
It replaces only the website/integration files and does not contain `models`,
`data`, or `reports`.

This edition uses broad animated liquid ribbons and one large left-side glass
sphere. The previous field of small floating background bubbles is removed.

## Replace the current frontend

1. Stop Flask and Vite using `Ctrl+C` in both terminals.
2. Extract this ZIP.
3. Copy everything inside the extracted folder.
4. Paste into `D:\support_ticket_project`.
5. Choose **Replace files in destination**.

## Run

Terminal 1:

```powershell
cd D:\support_ticket_project
.venv\Scripts\activate
python supportflow_app.py
```

Terminal 2:

```powershell
cd D:\support_ticket_project\frontend
npm install
npm run dev
```

Open `http://localhost:5173`, then press `Ctrl+Shift+R` once.
