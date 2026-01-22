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
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==============================
# HERO BG (base64 -> CSS variable)
# ==============================
def png_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

if os.path.exists(BANNER_PATH):
    b64 = png_to_base64(BANNER_PATH)
    st.markdown(
        f"<style>:root{{--hero-bg:url('data:image/png;base64,{b64}');}}</style>",
        unsafe_allow_html=True
    )

# ==============================
# INLINE POLISH (hapus shape kosong)
# ==============================
st.markdown("""
<style>
/* Ini yang bikin "shape putih kosong" sering muncul: elemen kosong di horizontal block */
div[data-testid="stHorizontalBlock"] > div:has(> div:empty){
  display:none !important;
}

/* garis pemisah tipis dalam buku */
.book-divider{
  height: 1px;
  width: 100%;
  margin: 12px 0 18px 0;
  background: linear-gradient(90deg,
    rgba(212,175,55,0.0),
    rgba(212,175,55,0.45),
    rgba(212,175,55,0.0)
  );
  opacity: .9;
}

/* judul section dalam buku */
.bookTitle{
  font-size: 1.15rem;
  font-weight: 900;
  color: rgba(31,41,55,0.92);
  margin: 0;
}
.bookSub{
  margin-top: 6px;
  color: rgba(31,41,55,0.62);
  font-size: 0.95rem;
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
# "BUKU" BESAR (1 CARD)
# ==============================
st.markdown('<div class="card">', unsafe_allow_html=True)

# Judul dalam buku (ini yang "ngisi" area atas biar ga kosong)
st.markdown("""
<div>
  <div class="bookTitle">📚 Panel Klasifikasi</div>
  <div class="bookSub">Masukkan gambar tokoh wayang di kiri, hasil prediksi akan muncul di kanan.</div>
  <div class="book-divider"></div>
</div>
""", unsafe_allow_html=True)

# ==============================
# 2 COLUMNS INSIDE THE BOOK
# ==============================
col_left, col_right = st.columns([1.25, 1], gap="large")

# ---------- LEFT: Upload & Preview ----------
with col_left:
    st.subheader("📤 Upload Gambar")

    uploaded = st.file_uploader(
        "Pilih file gambar (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    img = None
    if uploaded is None:
        st.caption("Silakan upload gambar tokoh wayang.")
    else:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)

# ---------- RIGHT: Prediction ----------
with col_right:
    st.subheader("📌 Hasil Prediksi")

    if img is None:
        st.caption("Hasil akan tampil setelah kamu upload gambar.")
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

        st.markdown("**Top-3 Prediksi:**")
        top3 = np.argsort(probs)[::-1][:3]
        for i, idx in enumerate(top3, start=1):
            st.progress(float(probs[idx]))
            st.write(f"**{i}. {CLASS_NAMES[idx]}** — {probs[idx]:.4f}")

st.markdown("</div>", unsafe_allow_html=True)
