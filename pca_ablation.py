"""
PCA Ablation Study — berapa n_components optimal?
Jalankan SETELAH data sudah di-split & di-augment (X_train.npy dst sudah ada).
Run: python pca_ablation.py
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
FIGURES_DIR   = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

N_COMPONENTS_LIST = [100, 200, 300, 400]
N_LATENCY_RUNS    = 100  # lebih banyak run = angka latency lebih stabil
LATENCY_TARGET_MS = 50   # batas baru yang kita set


def benchmark(model, scaler, pca, X_test):
    times = []
    for _ in range(N_LATENCY_RUNS):
        t0 = time.perf_counter()
        x_sc  = scaler.transform(X_test[:1])
        x_pca = pca.transform(x_sc)
        model.predict(x_pca)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def run():
    print("Loading data...")
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    X_test  = np.load(PROCESSED_DIR / "X_test.npy")
    y_test  = np.load(PROCESSED_DIR / "y_test.npy")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    svm_results = []
    knn_results = []

    for n in N_COMPONENTS_LIST:
        print(f"\n--- PCA n_components={n} ---")
        pca = PCA(n_components=n, random_state=42)
        X_tr = pca.fit_transform(X_train_sc)
        X_te = pca.transform(X_test_sc)
        var  = pca.explained_variance_ratio_.sum() * 100
        print(f"  Variance explained: {var:.1f}%")

        # SVM C=10 (best known)
        svm = SVC(kernel="rbf", C=10, random_state=42)
        svm.fit(X_tr, y_train)
        svm_f1  = f1_score(y_test, svm.predict(X_te), average="weighted")
        svm_lat = benchmark(svm, scaler, pca, X_test_sc)
        svm_results.append((n, var, svm_f1, svm_lat))
        print(f"  SVM   — F1: {svm_f1:.4f} | Latency: {svm_lat:.2f} ms")

        # KNN k=3
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_tr, y_train)
        knn_f1  = f1_score(y_test, knn.predict(X_te), average="weighted")
        knn_lat = benchmark(knn, scaler, pca, X_test_sc)
        knn_results.append((n, var, knn_f1, knn_lat))
        print(f"  KNN   — F1: {knn_f1:.4f} | Latency: {knn_lat:.2f} ms")

    # Print summary table
    print("\n" + "="*80)
    print(f"{'n_comp':>6} {'Var%':>6} | {'SVM F1':>7} {'SVM lat':>9} {'<50ms?':>7} | {'KNN F1':>7} {'KNN lat':>9} {'<50ms?':>7}")
    print("-"*80)
    for (n, var, sf, sl), (_, _, kf, kl) in zip(svm_results, knn_results):
        svm_ok = "✅" if sl < LATENCY_TARGET_MS else "❌"
        knn_ok = "✅" if kl < LATENCY_TARGET_MS else "❌"
        marker = " ← current" if n == 100 else ""
        print(f"{n:>6} {var:>5.1f}% | {sf:>7.4f} {sl:>8.2f}ms {svm_ok:>7} | {kf:>7.4f} {kl:>8.2f}ms {knn_ok:>7}{marker}")

    # Plot
    ns    = [r[0] for r in svm_results]
    vars_ = [r[1] for r in svm_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("PCA Ablation Study — n_components vs Performance", fontsize=13)

    # F1 Score
    axes[0].plot(ns, [r[2] for r in svm_results], "o-", label="SVM", color="#4e9af1")
    axes[0].plot(ns, [r[2] for r in knn_results], "s-", label="KNN", color="#f1a84e")
    axes[0].axvline(100, color="gray", linestyle="--", alpha=0.5, label="Current (100)")
    axes[0].set_xlabel("n_components")
    axes[0].set_ylabel("F1 Score (weighted)")
    axes[0].set_title("F1 Score")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Latency
    axes[1].plot(ns, [r[3] for r in svm_results], "o-", label="SVM", color="#4e9af1")
    axes[1].plot(ns, [r[3] for r in knn_results], "s-", label="KNN", color="#f1a84e")
    axes[1].axvline(100, color="gray", linestyle="--", alpha=0.5, label="Current (100)")
    axes[1].axhline(LATENCY_TARGET_MS, color="orange", linestyle="--", alpha=0.7, label=f"{LATENCY_TARGET_MS}ms target")
    axes[1].axhline(100, color="red", linestyle=":", alpha=0.4, label="100ms hard limit")
    axes[1].set_xlabel("n_components")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Inference Latency")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Variance explained
    axes[2].plot(ns, vars_, "D-", color="#a84ef1")
    axes[2].axvline(100, color="gray", linestyle="--", alpha=0.5, label="Current (100)")
    axes[2].set_xlabel("n_components")
    axes[2].set_ylabel("Variance Explained (%)")
    axes[2].set_title("PCA Variance Explained")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "pca_ablation.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    run()
