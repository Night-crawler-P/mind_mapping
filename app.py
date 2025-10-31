from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from flask import Flask, render_template, request, redirect, url_for, flash

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # Fallback if not installed yet; we guard usage


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")


def load_features_config() -> List[Dict[str, Any]]:
    features_config_path = os.path.join(os.path.dirname(__file__), "features.json")
    if not os.path.exists(features_config_path):
        return []
    with open(features_config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expecting a list of feature dicts with keys: name, label, type, default
    return data if isinstance(data, list) else []


class ModelWrapper:
    def __init__(self) -> None:
        self.model = None
        self.mode = None  # "pkl" or "py"

    def try_load(self) -> None:
        base_dir = os.path.dirname(__file__)
        pkl_path = os.path.join(base_dir, "model.pkl")
        py_path = os.path.join(base_dir, "model.py")

        if joblib is not None:
            # Prefer explicit model.pkl, otherwise fall back to first *.pkl in dir
            candidate_paths = []
            if os.path.exists(pkl_path):
                candidate_paths.append(pkl_path)
            else:
                for name in os.listdir(base_dir):
                    if name.lower().endswith(".pkl"):
                        candidate_paths.append(os.path.join(base_dir, name))
                # Stable order: prefer names containing "model"
                candidate_paths.sort(key=lambda p: ("model" not in os.path.basename(p).lower(), os.path.basename(p).lower()))
            if candidate_paths:
                self.model = joblib.load(candidate_paths[0])
                self.mode = "pkl"
                return

        if os.path.exists(py_path):
            # Dynamic import of user's model.py with a required `predict(features: Dict[str, float]) -> float` function
            import importlib.util

            spec = importlib.util.spec_from_file_location("user_model", py_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "predict"):
                    self.model = mod
                    self.mode = "py"
                    return

        self.model = None
        self.mode = None

    def is_ready(self) -> bool:
        return self.model is not None

    def get_feature_names(self) -> List[str]:
        if not self.is_ready():
            return []
        # scikit-learn models may expose feature_names_in_
        names = getattr(self.model, "feature_names_in_", None)
        if isinstance(names, (list, tuple)):
            return [str(n) for n in names]
        if getattr(self.model, "feature_names_in_", None) is not None:
            try:
                return [str(n) for n in list(self.model.feature_names_in_)]
            except Exception:
                pass
        return []

    def predict(self, ordered_feature_values: List[float], feature_map: Dict[str, Any]) -> float:
        if not self.is_ready():
            raise RuntimeError("Model not loaded")
        if self.mode == "pkl":
            # If model exposes feature_names_in_, try DataFrame with named columns to support categorical pipelines
            names = self.get_feature_names()
            if names:
                try:
                    import pandas as pd  # type: ignore
                    row = {name: feature_map.get(name) for name in names}
                    X_df = pd.DataFrame([row], columns=names)
                    y_pred = self.model.predict(X_df)
                    return float(y_pred[0])
                except Exception:
                    pass
            # Fallback: numeric-only ndarray in the order of features.json
            import numpy as np  # Local import
            X = np.array([ordered_feature_values], dtype=float)
            y_pred = self.model.predict(X)
            return float(y_pred[0])
        if self.mode == "py":
            # Expect a function: predict(features: Dict[str, float]) -> float
            return float(self.model.predict(feature_map))
        raise RuntimeError("Unknown model mode")


model_wrapper = ModelWrapper()
model_wrapper.try_load()


@app.route("/", methods=["GET"])
def index():
    features = load_features_config()
    model_ready = model_wrapper.is_ready()
    if (not features) and model_ready:
        inferred = model_wrapper.get_feature_names()
        if inferred:
            features = [{"name": n, "label": n.replace("_", " ").title(), "type": "number", "default": 0} for n in inferred]
    return render_template("index.html", features=features, model_ready=model_ready, prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    features = load_features_config()
    if not features:
        flash("features.json not found or empty. Add your feature definitions first.")
        return redirect(url_for("index"))

    if not model_wrapper.is_ready():
        model_wrapper.try_load()
        if not model_wrapper.is_ready():
            flash("Model not loaded. Place model.pkl (joblib) or model.py with predict().")
            return redirect(url_for("index"))

    values_in_order: List[float] = []
    feature_map: Dict[str, Any] = {}
    try:
        for feat in features:
            name = str(feat.get("name"))
            raw_val = request.form.get(name, "")
            ftype = (feat.get("type") or "number").lower()
            if ftype == "select":
                # keep categorical as string
                feature_map[name] = raw_val
            else:
                val = float(raw_val)
                values_in_order.append(val)
                feature_map[name] = val
    except ValueError:
        flash("Invalid input. Ensure all fields are numeric.")
        return redirect(url_for("index"))

    try:
        y_hat = model_wrapper.predict(values_in_order, feature_map)
    except Exception as e:  # pragma: no cover
        flash(f"Prediction error: {e}")
        return redirect(url_for("index"))

    return render_template("index.html", features=features, model_ready=True, prediction=y_hat)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)


