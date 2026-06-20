import numpy as np
import joblib
import json
import time
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
SPLIT_CONFIG = Path(__file__).parent.parent / "data" / "split" / "split_config.json"
MODELS_DIR = Path(__file__).parent.parent / "models"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["Fresh", "Rotten"]


def load_split_config():
    with open(SPLIT_CONFIG) as f:
        return json.load(f)


def load_presplit_data():
    """Load pre-split data from run_pipeline.py (augmented train, original test)."""
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    X_test  = np.load(PROCESSED_DIR / "X_test.npy")
    y_test  = np.load(PROCESSED_DIR / "y_test.npy")
    return X_train, y_train, X_test, y_test


def evaluate_model(model, X_test, y_test, display_name, file_slug):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    print(f"\n=== {display_name} ===")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {display_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{file_slug}.png", dpi=150)
    plt.close()

    return metrics


def benchmark_latency(model, scaler, pca, X_test, n_runs=100):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        x_sc = scaler.transform(X_test[:1])
        x_pca = pca.transform(x_sc)
        model.predict(x_pca)
        times.append((time.perf_counter() - start) * 1000)
    return np.mean(times), np.std(times)


def train(X_train=None, y_train=None, X_test=None, y_test=None):
    if X_train is None:
        print("Loading pre-split data...")
        X_train, y_train, X_test, y_test = load_presplit_data()

    config = load_split_config()
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"Train — Fresh: {(y_train==0).sum()}, Rotten: {(y_train==1).sum()}")
    print(f"Test  — Fresh: {(y_test==0).sum()},  Rotten: {(y_test==1).sum()}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print("\nFitting PCA (n_components=100)...")
    pca = PCA(n_components=100, random_state=config["random_state"])
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca = pca.transform(X_test_sc)
    explained = pca.explained_variance_ratio_.sum()
    print(f"Variance explained by 100 components: {explained*100:.1f}%")

    # Model definitions — LogReg untuned (baseline), others GridSearchCV
    model_configs = [
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(max_iter=1000, random_state=config["random_state"]),
            "param_grid": None,
            "file_slug": "logreg",
            "save_as": "baseline_logreg.pkl",
        },
        {
            "name": "KNN",
            "estimator": KNeighborsClassifier(),
            "param_grid": {"n_neighbors": [3, 5, 7, 9, 11]},
            "file_slug": "knn",
            "save_as": "knn_model.pkl",
        },
        {
            "name": "SVM",
            "estimator": SVC(kernel="rbf", random_state=config["random_state"]),
            "param_grid": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
            "file_slug": "svm",
            "save_as": "svm_model.pkl",
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(random_state=config["random_state"]),
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
            },
            "file_slug": "random_forest",
            "save_as": "rf_model.pkl",
        },
    ]

    results = {}

    for cfg in model_configs:
        name = cfg["name"]
        print(f"\n{'='*50}")
        print(f"Training: {name}")

        if cfg["param_grid"] is not None:
            gs = GridSearchCV(
                cfg["estimator"],
                cfg["param_grid"],
                cv=5,
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=1,
            )
            gs.fit(X_train_pca, y_train)
            best_model = gs.best_estimator_
            best_params = gs.best_params_
            print(f"Best params: {best_params}")
            display_name = f"{name} {best_params}"
        else:
            best_model = cfg["estimator"]
            best_model.fit(X_train_pca, y_train)
            display_name = f"{name} (untuned baseline)"
            best_params = {}

        metrics = evaluate_model(best_model, X_test_pca, y_test, display_name, cfg["file_slug"])
        lat_mean, lat_std = benchmark_latency(best_model, scaler, pca, X_test)
        print(f"Inference latency: {lat_mean:.2f} ± {lat_std:.2f} ms")

        results[name] = {
            "model": best_model,
            "metrics": metrics,
            "latency_ms": lat_mean,
            "latency_std": lat_std,
            "display_name": display_name,
            "best_params": best_params,
            "save_as": cfg["save_as"],
        }

    # Save shared components + individual models
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(pca, MODELS_DIR / "pca.pkl")
    for name, r in results.items():
        if r["save_as"]:
            joblib.dump(r["model"], MODELS_DIR / r["save_as"])
            print(f"Saved: models/{r['save_as']}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Model':<40} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Lat(ms)':>10}")
    print("-" * 80)
    for name, r in results.items():
        m = r["metrics"]
        print(
            f"{r['display_name']:<40} "
            f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['f1']:>6.3f} "
            f"{r['latency_ms']:>10.2f}"
        )

    return results


if __name__ == "__main__":
    results = train()
