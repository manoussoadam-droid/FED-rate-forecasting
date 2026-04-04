#!/usr/bin/env python3
"""Train TF-IDF + classifier; save artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from core.config import ARTIFACTS_DIR, MODEL_META_PATH, MODEL_PATH, USE_XGBOOST, VECTORIZER_PATH
from core.ingest import load_fomc, load_speaker, build_train_test_split


def main() -> None:
    fomc = load_fomc()
    speaker = load_speaker()
    train_df, test_df = build_train_test_split(fomc, speaker)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = train_df["document"].astype(str)
    y_train = train_df["decision"].astype(str)
    X_test = test_df["document"].astype(str)
    y_test = test_df["decision"].astype(str)

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)

    if USE_XGBOOST:
        import xgboost as xgb

        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_train)
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(le.classes_),
            max_depth=6,
            n_estimators=200,
            learning_rate=0.1,
            random_state=42,
            eval_metric="mlogloss",
        )
        model.fit(X_tr, y_tr_enc)
        model._decision_label_encoder = le  # type: ignore[attr-defined]
        kind = "xgboost"
    else:
        model = LogisticRegression(
            max_iter=10_000,
            class_weight="balanced",
            random_state=42,
            solver="saga",
        )
        model.fit(X_tr, y_train)
        kind = "logistic_regression"

    if USE_XGBOOST:
        le = getattr(model, "_decision_label_encoder")
        y_pred = le.inverse_transform(model.predict(X_te))
    else:
        y_pred = model.predict(X_te)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    if kind == "xgboost":
        classes_list = list(getattr(model, "_decision_label_encoder").classes_)
    else:
        classes_list = list(model.classes_)

    joblib.dump(
        {
            "model_type": kind,
            "classes": classes_list,
            "n_train": len(train_df),
            "n_test": len(test_df),
        },
        MODEL_META_PATH,
    )
    print(f"Saved: {VECTORIZER_PATH}, {MODEL_PATH}, {MODEL_META_PATH}")


if __name__ == "__main__":
    main()
