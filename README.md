# HeartCare Predictor — IBM Internship Project

This is a simple Flask-based web application demonstrating heart disease
prediction using a logistic regression model (scikit-learn). The app trains a
model on generated sample data if a pre-trained model is not found, and exposes
a web form to collect patient parameters and show prediction + probability.

Quick start (Windows PowerShell):

1. Create a virtual environment and activate it

```powershell
python -m venv .venv
; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
python -m pip install -r .\requirements.txt
```

3. Run the app

```powershell
python .\app.py
```

Open http://127.0.0.1:5000 in your browser.

Deployment

- For Heroku-like platforms: ensure `requirements.txt` and `Procfile` exist. Use `gunicorn` as the WSGI server and push to your platform.

Files added/important

- `app.py` — Flask app and routes
- `heart_disease_model.py` — model creation, train, save, load, predict
- `templates/` — HTML templates including `index.html`, `predict.html`, `about.html`
- `static/style.css` — basic styling
- `requirements.txt` — dependencies
- `Procfile` — gunicorn entry for deployment

If you want, I can:

- Run the app here to reproduce any errors and fix them.
- Add unit tests for `heart_disease_model.py`.
- Provide step-by-step Heroku or Railway deployment instructions.
