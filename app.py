import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Klasifikasi Tokoh Wayang",
    page_icon="🎭",
    layout="wide"
)

CLASS_NAMES = [
    "arjuna", "bagong", "bathara surya", "bathara wisnu", "gareng",
    "nakula", "petruk", "sadewa", "semar", "werkudara", "yudistira"
]

MODEL_PATH = "cnn_mobilenetv2_wayang_final.h5"
IMG_SIZE = (224, 224)

# ==============================
# LOAD CSS (assets/style.css)
# ==============================
def load_css(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS tidak ditemukan: {path}")

load_css("assets/style.css")

# ==============================
# WALLPAPER BACKGROUND (pakai assets/banner.png)
# ==============================
# NOTE: jangan tampilkan banner pakai st.image(), kita set jadi background
if os.path.exists("assets/banner.png"):
    st.markdown("""
    <style>
    /* Banner sebagai wallpaper */
    [data-testid="stAppViewContainer"]{
        background-image: url("assets/banner.png");
        background-size: cover;
        background-position: center top;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Biar konten kebaca: kasih overlay putih transparan + blur */
    .block-container{
        background: rgba(255,255,255,0.78);
        backdrop-filter: blur(6px);
        border-radius: 22px;
        padding: 22px !important;
        margin-top: 18px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.08);
    }

    /* Opsional: biar jarak atas rapi */
    .block-container > div:first-child{
        padding-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.warning("Banner tidak ditemukan: assets/banner.png (cek nama file/huruf besar-kecil)")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ==============================
# HERO HEADER (pakai class CSS)
# ==============================
st.markdown("""
<div class="hero">
  <span class="badge">🎭 Wayang Classification</span>
  <div class="heroTitle">Klasifikasi Tokoh Wayang</div>
  <div class="subtitle">Upload gambar wayang untuk melihat hasil klasifikasi (Top-3)</div>
</div>
""", unsafe_allow_html=True)

# ==============================
# MAIN LAYOUT
# ==============================
col_left, col_right = st.columns([1.4, 1], gap="large")

# ---------- LEFT ----------
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Gambar")

    uploaded = st.file_uploader(
        "Pilih file gambar (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded is None:
        st.info("Silakan upload gambar tokoh wayang.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT ----------
with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Hasil Prediksi")

    # preprocess
    img_resized = img.resize(IMG_SIZE)
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # predict
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx])

    # main prediction
    st.markdown(f"""
    <div class="metricRow">
        <div class="metric">
            <div class="label">Prediksi</div>
            <div class="value">{pred_label}</div>
        </div>
        <div class="metric">
            <div class="label">Confidence</div>
            <div class="value">{pred_conf:.4f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # top-3 (lebih visual)
    st.markdown("**Top-3 Prediksi:**")
    top3 = np.argsort(probs)[::-1][:3]
    for i, idx in enumerate(top3, start=1):
        st.progress(float(probs[idx]))
        st.write(f"**{i}. {CLASS_NAMES[idx]}** — {probs[idx]:.4f}")

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# FOOTER
# ==============================
st.markdown(
    "<div class='footer'>Model: CNN MobileNetV2 • Output: Top-3 Prediksi</div>",
    unsafe_allow_html=True
)
