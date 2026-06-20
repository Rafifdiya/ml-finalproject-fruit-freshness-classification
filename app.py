import time
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path
from PIL import Image
from src.features import extract_features

MODELS_DIR = Path(__file__).parent / "models"
CLASS_NAMES = ["Fresh", "Rotten"]
SUPPORTED_ITEMS = ["Apel", "Pisang", "Jeruk", "Wortel", "Timun", "Kentang", "Tomat"]

MODEL_CONFIGS = [
    {
        "key":          "svm",
        "display":      "SVM (RBF, C=10)",
        "short":        "SVM",
        "file":         "svm_model.pkl",
        "has_proba":    False,
        "rank":         1,
        "train_acc":    90.8,
        "train_f1":     90.8,
        "train_lat_ms": 0.93,
        "rank_color":   "#f59e0b",
        "desc": (
            "Akurasi tertinggi (90.8%) dengan latensi tercepat (0.93 ms). "
            "Kernel RBF menangkap pola non-linear di ruang PCA — ideal untuk deployment."
        ),
    },
    {
        "key":          "knn",
        "display":      "KNN (k=3)",
        "short":        "KNN",
        "file":         "knn_model.pkl",
        "has_proba":    True,
        "rank":         2,
        "train_acc":    90.0,
        "train_f1":     90.0,
        "train_lat_ms": 8.93,
        "rank_color":   "#94a3b8",
        "desc": (
            "Akurasi 90.0%. Prediksi berdasarkan k=3 tetangga terdekat di ruang PCA. "
            "Lebih lambat dari SVM karena harus hitung jarak ke seluruh training set tiap prediksi."
        ),
    },
    {
        "key":          "rf",
        "display":      "Random Forest",
        "short":        "RF",
        "file":         "rf_model.pkl",
        "has_proba":    True,
        "rank":         3,
        "train_acc":    86.5,
        "train_f1":     86.5,
        "train_lat_ms": 6.83,
        "rank_color":   "#cd7f32",
        "desc": (
            "Akurasi 86.5% dari voting ensemble pohon keputusan. "
            "Robust terhadap noise, namun latensi moderat karena melewati seluruh ensemble."
        ),
    },
    {
        "key":          "logreg",
        "display":      "Logistic Regression",
        "short":        "LogReg",
        "file":         "baseline_logreg.pkl",
        "has_proba":    True,
        "rank":         4,
        "train_acc":    74.4,
        "train_f1":     74.4,
        "train_lat_ms": 0.46,
        "rank_color":   "#6b7280",
        "desc": (
            "Baseline — akurasi terendah (74.4%). Model linear tidak cukup untuk menangkap "
            "pola non-linear fitur HOG + Histogram. Berguna sebagai pembanding saja."
        ),
    },
]

st.set_page_config(
    page_title="ML Freshness Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* background */
    .stApp { background-color: #0e1117; }
    section[data-testid="stSidebar"] { background-color: #161b27 !important; }

    /* hide white toolbar at top */
    header[data-testid="stHeader"] { background-color: #0e1117 !important; }
    [data-testid="stToolbar"]      { background-color: #0e1117 !important; }
    [data-testid="stDecoration"]   { display: none !important; }

    /* text */
    .stApp, .stMarkdown, p, li, span, label, div { color: #e2e8f0; }
    h1, h2, h3 { color: #f1f5f9 !important; }
    .stCaption, small { color: #94a3b8 !important; }

    /* metric cards */
    div[data-testid="stMetric"] {
        background: #1c2130;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricLabel"] > div  { color: #94a3b8 !important; }
    div[data-testid="stMetricValue"]        { color: #f1f5f9 !important; }
    div[data-testid="stMetricDelta"] > div  { color: #64748b !important; }

    /* expander */
    div[data-testid="stExpander"] {
        background: #1c2130;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
    }

    /* hide dataframe toolbar (search/download/fullscreen buttons) */
    [data-testid="stElementToolbar"] { display: none !important; }

    /* file uploader */
    [data-testid="stFileUploadDropzone"] * { background-color: #1c2130 !important; }
    [data-testid="stFileUploadDropzone"] {
        background-color: #1c2130 !important;
        border: 1px dashed #3a4460 !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploadDropzone"] p,
    [data-testid="stFileUploadDropzone"] span,
    [data-testid="stFileUploadDropzone"] small { color: #4ade80 !important; }
    [data-testid="stFileUploadDropzone"] button {
        background-color: #2a3650 !important;
        color: #e2e8f0 !important;
        border: 1px solid #3a4460 !important;
        border-radius: 6px !important;
    }

    /* scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-thumb { background: #2a2d3e; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    pca    = joblib.load(MODELS_DIR / "pca.pkl")
    return {cfg["key"]: joblib.load(MODELS_DIR / cfg["file"]) for cfg in MODEL_CONFIGS}, scaler, pca


def classify_all(img_path, models, scaler, pca):
    t0 = time.perf_counter()
    features = extract_features(img_path)
    feat_ms = (time.perf_counter() - t0) * 1000

    x_sc  = scaler.transform(features.reshape(1, -1))
    x_pca = pca.transform(x_sc)

    results = {}
    for cfg in MODEL_CONFIGS:
        model = models[cfg["key"]]
        t1 = time.perf_counter()

        if cfg["has_proba"]:
            proba      = model.predict_proba(x_pca)[0]
            label_idx  = int(model.predict(x_pca)[0])
            confidence = float(proba[label_idx])
        else:
            dec        = float(model.decision_function(x_pca)[0])
            label_idx  = 1 if dec > 0 else 0
            conf_rot   = 1.0 / (1.0 + np.exp(-dec))
            confidence = conf_rot if label_idx == 1 else 1.0 - conf_rot

        infer_ms = (time.perf_counter() - t1) * 1000
        results[cfg["key"]] = {
            "label":      CLASS_NAMES[label_idx],
            "confidence": confidence,
            "infer_ms":   round(infer_ms, 2),
        }

    return results, round(feat_ms, 2)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 ML Freshness")
    st.caption("Klasifikasi Kesegaran Sayur & Buah")
    st.divider()

    st.markdown("**Pipeline**")
    st.markdown(
        "HOG + Color Histogram → StandardScaler → PCA(100) → Model",
        help="Foreground mask diterapkan sebelum ekstraksi histogram warna."
    )
    st.divider()

    st.markdown("**Model (ranked by accuracy)**")
    for cfg in MODEL_CONFIGS:
        st.markdown(
            f"<span style='color:{cfg['rank_color']};font-weight:700;'>#{cfg['rank']}</span> "
            f"{cfg['display']} — **{cfg['train_acc']}%**",
            unsafe_allow_html=True,
        )
    st.divider()

    st.markdown("**Item yang Didukung**")
    st.markdown(" · ".join(SUPPORTED_ITEMS))
    st.divider()

    st.markdown("**Cara Penggunaan**")
    st.markdown(
        "1. Upload foto sayur/buah (JPG/PNG)\n"
        "2. Semua 4 model berjalan otomatis\n"
        "3. Bandingkan prediksi & keyakinan tiap model\n"
        "4. Lihat ranking di bagian bawah"
    )
    st.divider()
    st.caption("Kelompok 9 · COMP6577001 ML\nBINUS University 2025/2026")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Klasifikasi Kesegaran Sayur & Buah")
st.markdown("Upload satu gambar — **4 model** berjalan sekaligus dan hasilnya dibandingkan.")
st.divider()

models, scaler, pca = load_models()

uploaded = st.file_uploader(
    "Upload gambar (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="Gambar di-resize ke 128×128 px untuk ekstraksi fitur.",
)

if uploaded is None:
    st.stop()

# ── Image + consensus ─────────────────────────────────────────────────────────
col_img, col_info = st.columns([1, 1], gap="large")

with col_img:
    st.subheader("Gambar Input")
    image = Image.open(uploaded)
    st.image(image, use_column_width=True)
    st.caption(f"{uploaded.name} · {image.size[0]}×{image.size[1]} px")

tmp_path = Path("tmp_upload.jpg")
with open(tmp_path, "wb") as f:
    f.write(uploaded.getbuffer())

try:
    all_results, feat_ms = classify_all(tmp_path, models, scaler, pca)
except Exception as e:
    st.error(f"Gagal memproses gambar: {e}")
    tmp_path.unlink(missing_ok=True)
    st.stop()

tmp_path.unlink(missing_ok=True)

with col_info:
    st.subheader("Info Ekstraksi Fitur")
    st.metric("Ekstraksi Fitur", f"{feat_ms:.2f} ms", "HOG + Color Histogram · shared semua model")

# ── 4 Model cards ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("Hasil Per Model")

cols = st.columns(4, gap="small")
for col, cfg in zip(cols, MODEL_CONFIGS):
    res       = all_results[cfg["key"]]
    label_id  = "Segar" if res["label"] == "Fresh" else "Tidak Segar"
    pred_color = "#4ade80" if res["label"] == "Fresh" else "#fb923c"
    conf_pct  = res["confidence"] * 100
    total_ms  = round(feat_ms + res["infer_ms"], 2)
    svm_note  = "*" if not cfg["has_proba"] else ""

    with col:
        st.markdown(f"""
        <div style="background:#1c2130;border-radius:12px;padding:18px 14px;
                    border-top:3px solid {cfg['rank_color']};text-align:center;height:100%;">
            <div style="color:{cfg['rank_color']};font-size:0.75rem;font-weight:700;
                        letter-spacing:0.05em;margin-bottom:4px;">
                #{cfg['rank']} &nbsp;·&nbsp; {cfg['short']}
            </div>
            <div style="color:#94a3b8;font-size:0.7rem;margin-bottom:12px;">
                {cfg['train_acc']}% train acc
            </div>
            <div style="color:{pred_color};font-size:1.15rem;font-weight:700;margin-bottom:4px;">
                {label_id}
            </div>
            <div style="color:{pred_color};font-size:2rem;font-weight:800;line-height:1;">
                {conf_pct:.1f}%{svm_note}
            </div>
            <div style="color:#64748b;font-size:0.7rem;margin-bottom:14px;">keyakinan</div>
            <div style="border-top:1px solid #2a2d3e;padding-top:10px;
                        color:#64748b;font-size:0.72rem;">
                {res['infer_ms']} ms inferensi &nbsp;·&nbsp; {total_ms} ms total
            </div>
        </div>
        """, unsafe_allow_html=True)

st.caption("* SVM: keyakinan via sigmoid(decision_function), bukan probabilitas kalibrasi.")

# ── Performance table ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Perbandingan Performa Model")
st.caption("Diukur pada test set (20% data). Latency = inferensi saja, tanpa ekstraksi fitur (~7 ms).")

df = pd.DataFrame([
    {
        "Rank":         f"#{cfg['rank']}",
        "Model":        cfg["display"],
        "Accuracy (%)": cfg["train_acc"],
        "F1 Score (%)": cfg["train_f1"],
        "Latency (ms)": cfg["train_lat_ms"],
    }
    for cfg in MODEL_CONFIGS
])

st.dataframe(
    df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Rank":         st.column_config.TextColumn(width=70),
        "Model":        st.column_config.TextColumn(width="medium"),
        "Accuracy (%)": st.column_config.NumberColumn("Accuracy", format="%.1f %%", width=110),
        "F1 Score (%)": st.column_config.NumberColumn("F1 Score", format="%.1f %%", width=110),
        "Latency (ms)": st.column_config.NumberColumn("Latency",  format="%.2f ms", width=110),
    },
)

# ── Model analysis ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Analisis Model")

for cfg in MODEL_CONFIGS:
    with st.expander(f"#{cfg['rank']} · {cfg['display']} — {cfg['train_acc']}% acc · {cfg['train_lat_ms']} ms"):
        st.markdown(cfg["desc"])
