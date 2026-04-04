"""Load trained vectorizer + model; predict + simple explanation via coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from scipy import sparse

from core.config import MODEL_PATH, VECTORIZER_PATH


@dataclass
class LoadedModel:
    vectorizer: Any
    model: Any
    model_type: str


def load_predictor() -> LoadedModel | None:
    if not VECTORIZER_PATH.is_file() or not MODEL_PATH.is_file():
        return None
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    mt = "xgboost" if hasattr(model, "get_booster") else "logistic_regression"
    return LoadedModel(vectorizer=vectorizer, model=model, model_type=mt)


def predict_decision(text: str, clip_chars: int = 100_000) -> dict[str, Any]:
    pred = load_predictor()
    if pred is None:
        return {
            "prediction": None,
            "probabilities": None,
            "top_features": None,
            "error": "Model artifacts missing. Run: python scripts/train_model.py",
        }
    t = text[:clip_chars]
    X = pred.vectorizer.transform([t])

    if pred.model_type == "xgboost":
        le = getattr(pred.model, "_decision_label_encoder", None) or getattr(
            pred.model, "_fednlp_label_encoder", None
        )
        if le is None:
            return {"prediction": None, "probabilities": None, "top_features": None, "error": "Missing label encoder on model"}
        proba = pred.model.predict_proba(X)[0]
        classes = list(le.classes_)
        pred_idx = int(np.argmax(proba))
        label = classes[pred_idx]
        probs = {classes[i]: float(proba[i]) for i in range(len(classes))}
        return {
            "prediction": label,
            "probabilities": probs,
            "top_features": None,
            "error": None,
        }

    # LogisticRegression (binary or multiclass one-vs-rest)
    proba = pred.model.predict_proba(X)[0]
    classes = list(pred.model.classes_)
    pred_idx = int(np.argmax(proba))
    label = classes[pred_idx]
    probs = {classes[i]: float(proba[i]) for i in range(len(classes))}

    top_features = _top_lr_features(pred.vectorizer, pred.model, X, label, classes)
    return {
        "prediction": label,
        "probabilities": probs,
        "top_features": top_features,
        "error": None,
    }


def _top_lr_features(vectorizer, model, X: sparse.csr_matrix, predicted_label: str, classes: list[str], k: int = 12) -> list[dict[str, Any]]:
    """For multinomial LR, use coef row for predicted class."""
    try:
        feat_names = vectorizer.get_feature_names_out()
    except Exception:
        return []
    if not hasattr(model, "coef_"):
        return []
    coef = model.coef_
    if coef.ndim == 1:
        w = coef
    else:
        idx = classes.index(predicted_label) if predicted_label in classes else int(np.argmax(model.predict_proba(X)[0]))
        w = coef[idx]
    x = X.toarray().ravel()
    contrib = np.asarray(w).ravel() * x
    top_idx = np.argsort(-np.abs(contrib))[:k]
    return [
        {
            "token": str(feat_names[i]),
            "contribution": float(contrib[i]),
        }
        for i in top_idx
        if contrib[i] != 0
    ]
