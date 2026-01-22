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

CSS_PATH = "assets/style.css"
BANNER_PATH = "assets/banner.png"

# load css
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# inject banner base64
def png_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

if os.path.exists(BANNER_PATH):
    b64 = png_to_b64(BANNER_PATH)
    st.markdown(f"<style>:root{{--hero-bg:url('data:image/png;base64,{b64}');}}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# HERO
st.markdown("""
<div class="hero">
  <div class="heroInner">
    <span class="badge">🎭 Wayang Classification</span>
    <div class="heroTitle">Klasifikasi Tokoh Wayang</div>
    <div class="subtitle">Upload gambar wayang untuk melihat hasil klasifikasi (Top-3)</div>
  </div>
</div>
""", unsafe_allow_html=True)

# BOOK (1 besar)
st.markdown('<div class="book">', unsafe_allow_html=True)
st.markdown('<div class="ornament"></div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.25, 1], gap="large")

with col_left:
    st.markdown('<div class="sectionTitle">📤 Upload Gambar</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Pilih file gambar (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    img = None
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)
    else:
        st.caption("Upload gambar untuk memulai.")

with col_right:
    st.markdown('<div class="sectionTitle">📌 Hasil Prediksi</div>', unsafe_allow_html=True)

    if img is None:
        st.caption("Hasil prediksi akan muncul di sini.")
    else:
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
