# Klasifikasi Kesegaran Sayur & Buah
**COMP6577001 Machine Learning | BINUS University | Kelompok 9**

Sistem klasifikasi otomatis untuk membedakan sayur dan buah **Segar (Fresh)** vs **Tidak Segar (Rotten)** menggunakan 4 model classical machine learning.

---

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

## Limitasi

- **Domain shift background:** Dataset berisi foto e-commerce dengan background putih. Foreground mask hanya mengecualikan pixel putih (S<30 & V>180) — foto dengan background non-putih (meja kayu, dinding, dll) dapat memengaruhi akurasi color histogram.
- **7 item terbatas:** Model hanya dilatih untuk Apple, Banana, Orange, Carrot, Cucumber, Potato, Tomato. Item di luar daftar ini tidak dapat diklasifikasikan dengan andal.
- **Tidak ada segmentasi objek:** Sistem tidak dapat mendeteksi lokasi buah/sayur dalam gambar — seluruh frame diasumsikan berisi satu objek.
- **Akurasi klasikal ML:** Akurasi tertinggi 91.1% (SVM) — beberapa misklasifikasi pada gambar dengan pencahayaan ekstrem atau kondisi setengah busuk adalah hal yang normal.

---

## Future Work

- **Background removal adaptif:** Implementasi segmentasi objek (GrabCut atau model ringan seperti `rembg`) agar color histogram tidak terkontaminasi background non-putih.
- **Ekspansi item:** Tambah lebih banyak jenis sayur dan buah ke dataset training.
- **Deteksi multi-objek:** Integrasikan object detection (YOLO atau sliding window) agar dapat memproses foto dengan beberapa buah sekaligus.
- **Kalibrasi confidence SVM:** Gunakan `CalibratedClassifierCV` agar probabilitas SVM lebih akurat dibanding sigmoid(decision_function).
- **Deployment cloud:** Deploy ke Streamlit Cloud atau HuggingFace Spaces agar dapat diakses tanpa instalasi lokal.

---

## Catatan

- `data/raw/` dan `data/processed/` tidak di-commit (dataset besar)
- `models/rf_model.pkl` tidak di-commit (>100MB) — jalankan `run_pipeline.py` untuk generate ulang
- Training dilakukan di Google Colab menggunakan dataset dengan augmentasi brightness (3× data training)
