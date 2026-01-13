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
    layout="wide"  # <-- landscape
)

CLASS_NAMES = [
    "arjuna", "bagong", "bathara surya", "bathara wisnu", "gareng",
    "nakula", "petruk", "sadewa", "semar", "werkudara", "yudistira"
]

MODEL_PATH = "cnn_mobilenetv2_wayang_final.h5"
IMG_SIZE = (224, 224)

# ==============================
# STYLE (clean + wide)
# ==============================
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 1.5rem; max-width: 1200px;}
      .title {font-size: 2.1rem; font-weight: 900; margin-bottom: 0.25rem;}
      .subtitle {color: #6b7280; margin-bottom: 1.25rem;}
      .card {
        border: 1px solid rgba(0,0,0,0.07);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        background: #ffffff;
      }
      .metricRow {display:flex; gap:10px; flex-wrap:wrap; margin: 10px 0 5px 0;}
      .metricPill {
        padding: 10px 12px;
        border-radius: 14px;
        border: 1px solid rgba(0,0,0,0.07);
        background: #fafafa;
        min-width: 200px;
      }
      .metricPill .k {color:#6b7280; font-size:0.9rem;}
      .metricPill .v {font-size:1.15rem; font-weight:800;}
      .small {color:#6b7280; font-size:0.9rem;}
      .top3 li {margin-bottom: 6px;}
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ==============================
# HEADER
# ==============================
st.markdown('<div class="title">🎭 Klasifikasi Tokoh Wayang</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload gambar wayang → sistem memprediksi kelas (Top-3) menggunakan CNN MobileNetV2.</div>',
    unsafe_allow_html=True
)

# ==============================
# LAYOUT: 2 columns (image big on left, results on right)
# ==============================
col_left, col_right = st.columns([1.35, 1], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload Gambar")

    uploaded = st.file_uploader("Pilih file (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.markdown('<div class="small">Tips: pastikan wayang terlihat jelas dan tidak blur.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    img = Image.open(uploaded).convert("RGB")

    # TAMPILKAN GAMBAR LEBIH BESAR (lebar kolom)
    st.image(img, caption="Preview gambar", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Hasil Prediksi")

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

    # metrics
    st.markdown(
        f"""
        <div class="metricRow">
          <div class="metricPill">
            <div class="k">Prediksi</div>
            <div class="v">{pred_label}</div>
          </div>
          <div class="metricPill">
            <div class="k">Confidence</div>
            <div class="v">{pred_conf:.4f}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Top-3
    top3 = np.argsort(probs)[::-1][:3]
    st.markdown("<div class='small' style='margin-top:10px;'><b>Top-3 Prediksi</b></div>", unsafe_allow_html=True)
    st.markdown("<ul class='top3'>", unsafe_allow_html=True)
    for rank, idx in enumerate(top3, start=1):
        st.markdown(f"<li><b>{rank}. {CLASS_NAMES[idx]}</b> — {float(probs[idx]):.4f}</li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="small" style="text-align:center; margin-top: 14px;">Model: CNN MobileNetV2 • Output: Top-3 Prediksi</div>', unsafe_allow_html=True)
