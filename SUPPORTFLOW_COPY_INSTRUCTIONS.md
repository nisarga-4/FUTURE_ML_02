# Add SupportFlow to the existing project

This package is designed to be copied into:

```text
D:\support_ticket_project
```

It intentionally contains no `models` or `data` folder. Your trained models,
datasets, reports and existing Python programs will remain untouched.

## Copy

1. Stop the current Flask server with `Ctrl+C`.
2. Extract `SupportFlow_Upgrade_Only.zip`.
3. Select everything inside the extracted folder.
4. Copy and paste it into `D:\support_ticket_project`.
5. If Windows asks about the `frontend` folder, choose **Merge/Replace files**.

## Run

Open `D:\support_ticket_project` in VS Code and enter:

```powershell
.venv\Scripts\activate
pip install -r requirements_supportflow.txt
python supportflow_app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The optimized React build is included, so npm is not required simply to run
the website. Use the following only after editing files inside `frontend/src`:

```powershell
cd frontend
npm install
npm run build
cd ..
python supportflow_app.py
```

## Model files expected

The application uses the models already present at:

```text
models\category_advanced\category_model_advanced.joblib
models\priority_business\priority_business_model.joblib
```

