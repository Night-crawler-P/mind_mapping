House Price Predictor Web App
=============================

Run locally
-----------

1. Create and activate a virtual environment (recommended).
   - Windows PowerShell:
     - python -m venv .venv
     - .venv\\Scripts\\Activate.ps1
2. Install dependencies:
   - pip install -r requirements.txt
3. Put your model in the project folder (same directory as app.py) using one of:
   - model.pkl: a joblib-dumped scikit-learn estimator with .predict(X)
   - model.py: a Python file exposing `predict(features: Dict[str, float]) -> float`
4. Define your input fields in features.json. Order must match your model training order.
5. Start the app:
   - set FLASK_SECRET_KEY=some-secret (optional)
   - python app.py
6. Open http://localhost:5000

Notes
-----
- If using model.pkl, the app passes a single row shaped like [[f1, f2, ...]] in the order of features.json.
- If model exposes `feature_names_in_` (common for sklearn pipelines), the app will send a pandas DataFrame with named columns so categorical encoders (e.g., OneHotEncoder) work.
- If using model.py, implement a `predict(dict)` function that returns a numeric price.
- Edit features.json labels and defaults to fit your model.


