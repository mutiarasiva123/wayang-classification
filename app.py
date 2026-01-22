import os
import base64
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

# ==============================
# CONSTANTS
# ==============================
CLASS_NAMES = [
    "arjuna", "bagong", "bathara surya", "bathara wisnu", "gareng",
    "nakula", "petruk", "sadewa", "semar", "werkudara", "yudistira"
]
MODEL_PATH = "cnn_mobilenetv2_wayang_final.h5"
IMG_SIZE = (224, 224)

CSS_PATH = "assets/style.css"
BANNER_PATH = "assets/banner.png"

# ==============================
# LOAD CSS
# ==============================
def load_css(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(CSS_PATH)

# ==============================
# INJECT BANNER AS CSS VARIABLE (BASE64)
# ==============================
def png_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

if os.path.exists(BANNER_PATH):
    hero_bg = png_to_data_uri(BANNER_PATH)
    st.markdown(
        f"<style>:root{{--hero-bg:url('{hero_bg}');}}</style>",
        unsafe_allow_html=True
    )

# ==============================
# INLINE CSS: divider gold + small polish
# ==============================
st.markdown("""
<style>
/* Divider tipis gold di tengah kolom (di dalam big card) */
.v-divider{
  width: 1px;
  background: linear-gradient(180deg,
    rgba(212,175,55,0.05),
    rgba(212,175,55,0.45),
    rgba(212,175,55,0.05)
  );
  border-radius: 999px;
  margin: 8px 0;
}

/* Halusin caption biar clean */
.small-muted{
  color: rgba(31,41,55,0.60);
  font-size: 0.92rem;
  margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ==============================
# HERO
# ==============================
st.markdown("""
<div class="hero">
  <div class="heroInner">
    <span class="badge">🎭 Wayang Classification</span>
    <div class="heroTitle">Klasifikasi Tokoh Wayang</div>
    <div class="subtitle">Upload gambar wayang untuk melihat hasil klasifikasi (Top-3)</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ==============================
# BIG CARD (Upload + Result in one panel)
# ==============================
st.markdown('<div class="card">', unsafe_allow_html=True)

# 3 kolom: kiri | divider | kanan
col_left, col_div, col_right = st.columns([1.25, 0.04, 1], gap="large")

# ---------- LEFT ----------
with col_left:
    st.subheader("📤 Upload Gambar")

    uploaded = st.file_uploader(
        "Pilih file gambar (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded is None:
        st.markdown("<div class='small-muted'>Silakan upload gambar tokoh wayang.</div>", unsafe_allow_html=True)
        img = None
    else:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)

# ---------- DIVIDER ----------
with col_div:
    st.markdown("<div class='v-divider' style='height: 100%;'></div>", unsafe_allow_html=True)

# ---------- RIGHT ----------
with col_right:
    st.subheader("📌 Hasil Prediksi")

    if img is None:
        st.markdown("<div class='small-muted'>Hasil prediksi akan muncul setelah kamu upload gambar.</div>", unsafe_allow_html=True)
    else:
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

        # top-3
        st.markdown("**Top-3 Prediksi:**")
        top3 = np.argsort(probs)[::-1][:3]
        for i, idx in enumerate(top3, start=1):
            st.progress(float(probs[idx]))
            st.write(f"**{i}. {CLASS_NAMES[idx]}** — {probs[idx]:.4f}")

st.markdown("</div>", unsafe_allow_html=True)
