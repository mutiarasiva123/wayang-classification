import os
import base64
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Klasifikasi Tokoh Wayang", page_icon="🎭", layout="wide")

CLASS_NAMES = [
    "arjuna", "bagong", "bathara surya", "bathara wisnu", "gareng",
    "nakula", "petruk", "sadewa", "semar", "werkudara", "yudistira"
]
MODEL_PATH = "cnn_mobilenetv2_wayang_final.h5"
IMG_SIZE = (224, 224)

def load_css(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS tidak ditemukan: {path}")

load_css("assets/style.css")

# ====== FORCE REMOVE TOP GAP + HERO FIX ======
def img_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"data:image/png;base64,{b64}"

banner_uri = None
if os.path.exists("assets/banner.png"):
    banner_uri = img_to_data_uri("assets/banner.png")

st.markdown(f"""
<style>
/* benerin jarak atas streamlit */
.block-container {{
  padding-top: 0rem !important;
  margin-top: -48px !important;
  max-width: 1300px;
}}

/* HERO yang gak kepotong + pasti pakai banner */
.hero {{
  position: relative;
  width: 100%;
  min-height: 240px;              /* biar gak kepotong */
  border-radius: 0 0 28px 28px;
  padding: 56px 36px;
  margin: 0 0 26px 0;
  overflow: hidden;
  background-image: url("{banner_uri if banner_uri else ''}");
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  box-shadow: 0 14px 35px rgba(0,0,0,0.14);
}}

.hero::before {{
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(rgba(0,0,0,0.25), rgba(0,0,0,0.55));
}}

.hero > * {{
  position: relative;
  z-index: 2;
}}

.heroTitle {{
  font-size: 2.65rem;
  font-weight: 900;
  margin: 10px 0 0 0;
  color: #fff;
}}

.subtitle {{
  color: rgba(255,255,255,0.92);
  margin-top: 10px;
  font-size: 1.05rem;
}}

.badge {{
  display: inline-block;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.18);
  border: 1px solid rgba(255,255,255,0.38);
  color: #fff;
  font-weight: 800;
  font-size: 0.85rem;
  backdrop-filter: blur(6px);
}}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ===== HERO =====
st.markdown("""
<div class="hero">
  <span class="badge">🎭 Wayang Classification</span>
  <div class="heroTitle">Klasifikasi Tokoh Wayang</div>
  <div class="subtitle">Upload gambar wayang untuk melihat hasil klasifikasi (Top-3)</div>
</div>
""", unsafe_allow_html=True)

# ===== MAIN LAYOUT =====
col_left, col_right = st.columns([1.4, 1], gap="large")

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

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Hasil Prediksi")

    img_resized = img.resize(IMG_SIZE)
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx])

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

    st.markdown("**Top-3 Prediksi:**")
    top3 = np.argsort(probs)[::-1][:3]
    for i, idx in enumerate(top3, start=1):
        st.progress(float(probs[idx]))
        st.write(f"**{i}. {CLASS_NAMES[idx]}** — {probs[idx]:.4f}")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Model: CNN MobileNetV2 • Output: Top-3 Prediksi</div>", unsafe_allow_html=True)
