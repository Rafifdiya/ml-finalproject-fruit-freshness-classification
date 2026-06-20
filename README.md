# Klasifikasi Kesegaran Sayur & Buah
**COMP6577001 Machine Learning | BINUS University | Kelompok 9**

Sistem klasifikasi otomatis untuk membedakan sayur dan buah **Segar (Fresh)** vs **Tidak Segar (Rotten)** menggunakan 4 model classical machine learning.

---

## Hasil Model

| Model | Accuracy | F1 Score | Latency |
|-------|----------|----------|---------|
| **SVM (RBF, C=10)** | **91.1%** | **91.1%** | **3.61 ms** |
| KNN (k=3) | 90.0% | 90.0% | 11.42 ms |
| Random Forest (n=200) | 85.8% | 85.8% | 25.87 ms |
| Logistic Regression (baseline) | 75.3% | 75.2% | 2.54 ms |

- Dataset: 8,387 gambar × 7 item (Apple, Banana, Orange, Carrot, Cucumber, Potato, Tomato)
- Feature extraction: HOG 128×128 + Color Histogram (HSV) + foreground mask
- Dimensionality reduction: StandardScaler → PCA(100)
- Semua model memenuhi target latency **< 100 ms** ✅

---

## Instalasi

```bash
pip install -r requirements.txt
```

## Jalankan Aplikasi

```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## Struktur Folder

```
├── app.py                  # Streamlit app — upload gambar → 4 model prediksi sekaligus
├── run_pipeline.py         # Feature extraction + training (jalankan di Google Colab)
├── pca_ablation.py         # Eksperimen PCA n_components vs accuracy & latency
├── requirements.txt
├── src/
│   ├── features.py         # HOG + Color Histogram pipeline
│   ├── train.py            # Training 4 model dengan GridSearchCV
│   └── predict.py          # Inference function
├── models/                 # Model .pkl (rf_model.pkl tidak di-commit, >100MB)
│   ├── svm_model.pkl
│   ├── knn_model.pkl
│   ├── baseline_logreg.pkl
│   ├── scaler.pkl
│   └── pca.pkl
├── figures/                # Visualisasi EDA, confusion matrix, ROC-AUC, LIME
└── data/
    └── split/
        └── split_config.json
```

---

## Catatan

- `data/raw/` dan `data/processed/` tidak di-commit (dataset besar)
- `models/rf_model.pkl` tidak di-commit (>100MB) — jalankan `run_pipeline.py` untuk generate ulang
- Training dilakukan di Google Colab menggunakan dataset dengan augmentasi brightness (3× data training)
