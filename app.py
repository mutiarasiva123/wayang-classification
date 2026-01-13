import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Klasifikasi Tokoh Wayang",
    page_icon="🎭",
    layout="centered"
)

# Urutan kelas harus sama seperti training
CLASS_NAMES = [
    "arjuna", "bagong", "bathara surya", "bathara wisnu", "gareng",
    "nakula", "petruk", "sadewa", "semar", "werkudara", "yudistira"
]

MODEL_PATH = "cnn_mobilenetv2_wayang_final.h5"
IMG_SIZE = (224, 224)

# ==============================
# STYLE (simple modern)
# ==============================
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 900px;}
      .title {font-size: 2.0rem; font-weight: 800; margin-bottom: 0.2rem;}
      .subtitle {color: #6b7280; margin-bottom: 1.2rem;}
      .card {
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.04);
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(6px);
      }
      .metric {
        display: inline-block;
        padding: 10px 12px;
        border-radius: 14px;
        border: 1px solid rgba(0,0,0,0.06);
        background: rgba(250,250,250,0.9);
        margin-right: 10px;
      }
      .metric b {font-size: 1.05rem;}
      .small {color: #6b7280; font-size: 0.9rem;}
      .divider {height: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# LOAD MODEL (cached)
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ==============================
# UI
# ==============================
st.markdown('<div class="title">🎭 Klasifikasi Tokoh Wayang</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload gambar wayang, lalu sistem memprediksi tokoh (Top-3) menggunakan CNN MobileNetV2.</div>',
    unsafe_allow_html=True
)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.markdown('<div class="small">Tips: gunakan gambar yang jelas, tidak blur, dan objek wayang terlihat dominan.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # Show image
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_container_width=True)

    # Preprocess
    img_resized = img.resize(IMG_SIZE)
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx])

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("Hasil Prediksi")

    st.markdown(
        f"""
        <div class="metric"><span class="small">Prediksi</span><br><b>{pred_label}</b></div>
        <div class="metric"><span class="small">Confidence</span><br><b>{pred_conf:.4f}</b></div>
        """,
        unsafe_allow_html=True
    )

    # Top-3
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("Top-3 Prediksi")

    top3 = np.argsort(probs)[::-1][:3]
    top3_labels = [CLASS_NAMES[i] for i in top3]
    top3_scores = [float(probs[i]) for i in top3]

    for rank, (lbl, sc) in enumerate(zip(top3_labels, top3_scores), start=1):
        st.write(f"**{rank}. {lbl}** — {sc:.4f}")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer kecil
st.markdown(
    '<div class="small" style="text-align:center; margin-top: 18px;">Model: CNN MobileNetV2 • Output: Top-3 Prediksi</div>',
    unsafe_allow_html=True
)
