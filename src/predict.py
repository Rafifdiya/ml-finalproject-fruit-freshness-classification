import time
import numpy as np
import joblib
from pathlib import Path
from .features import extract_features

MODELS_DIR = Path(__file__).parent.parent / "models"

_knn = None
_scaler = None
_pca = None


def _load_models():
    global _knn, _scaler, _pca
    if _knn is None:
        _knn = joblib.load(MODELS_DIR / "knn_model.pkl")
        _scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        _pca = joblib.load(MODELS_DIR / "pca.pkl")


def predict(img_path) -> dict:
    _load_models()

    t0 = time.perf_counter()
    features = extract_features(img_path)
    feat_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    x = _scaler.transform(features.reshape(1, -1))
    x = _pca.transform(x)
    label_idx = _knn.predict(x)[0]
    proba = _knn.predict_proba(x)[0]
    infer_ms = (time.perf_counter() - t1) * 1000

    label = "Fresh" if label_idx == 0 else "Rotten"
    confidence = float(proba[label_idx])
    total_ms = feat_ms + infer_ms

    return {
        "label": label,
        "label_id": int(label_idx),
        "confidence": confidence,
        "probabilities": {"Fresh": float(proba[0]), "Rotten": float(proba[1])},
        "latency_ms": {
            "feature_extraction": round(feat_ms, 2),
            "inference": round(infer_ms, 2),
            "total": round(total_ms, 2),
        },
    }
