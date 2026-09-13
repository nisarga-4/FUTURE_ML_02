# GitHub submission steps

Run these commands from the project directory after confirming that the
application works locally.

```powershell
git init
git add .
git commit -m "Build SupportFlow AI ticket decision system"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/supportflow-ai.git
git push -u origin main
```

Before pushing:

- Replace `YOUR_USERNAME` with your GitHub username.
- Confirm raw CSVs and `.joblib` models are not staged.
- Add two real screenshots to a `docs/screenshots` folder and reference them in
  `README.md`.
- Never commit customer names, email addresses or private support text.

Useful final commit sequence:

```powershell
git add .
git commit -m "Add trained model evaluation reports"

git add .
git commit -m "Add React liquid-glass dashboard"

git add .
git commit -m "Complete documentation and project presentation"
```
