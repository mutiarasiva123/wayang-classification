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

# Urutan kelas harus sama seperti training
CLASS_NAMES = [
    "arjuna", "bagong", "bathara surya", "bathara wisnu", "gareng",
    "nakula", "petruk", "sadewa", "semar", "werkudara", "yudistira"
]

MODEL_PATH = "cnn_mobilenetv2_wayang_final.h5"
IMG_SIZE = (224, 224)

# ==============================
# CUSTOM STYLE (CLEAN & WIDE) + HIDE WHITE SHAPES
# ==============================
st.markdown("""
<style>
/* Layout */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem;
    max-width: 1300px;
}

/* Hide Streamlit top decorations / toolbars (removes white rounded shapes) */
div[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"] {
    display: none !important;
}

/* If any empty text inputs get rendered, hide them */
div[data-testid="stTextInput"],
div[data-testid="stTextInput"] > div,
input[type="text"] {
    display: none !important;
}

/* Typography */
h1 {
    font-size: 2.4rem;
    font-weight: 900;
    margin-bottom: 0.25rem;
}
.subtitle {
    color: #6b7280;
    margin-bottom: 1.5rem;
    font-size: 1rem;
}

/* Cards */
.card {
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 18px;
    padding: 20px;
    background: #ffffff;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
}

/* Metrics */
.metricRow {
    display: flex;
    gap: 14px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}
.metric {
    padding: 12px 16px;
    border-radius: 14px;
    border: 1px solid rgba(0,0,0,0.08);
    background: #fafafa;
    min-width: 200px;
}
.metric .label {
    font-size: 0.9rem;
    color: #6b7280;
}
.metric .value {
    font-size: 1.2rem;
    font-weight: 800;
}

/* Top-3 list */
.top3 li {
    margin-bottom: 6px;
    font-size: 1rem;
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
# HEADER
# ==============================
st.title("🎭 Klasifikasi Tokoh Wayang")
st.markdown('<div class="subtitle">Upload gambar wayang untuk melihat hasil klasifikasi</div>', unsafe_allow_html=True)

# ==============================
# MAIN LAYOUT
# ==============================
col_left, col_right = st.columns([1.4, 1], gap="large")

# ---------- LEFT: IMAGE ----------
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload Gambar")

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

# ---------- RIGHT: RESULT ----------
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
    st.markdown("<ul class='top3'>", unsafe_allow_html=True)
    for i, idx in enumerate(top3, start=1):
        st.markdown(f"<li><b>{i}. {CLASS_NAMES[idx]}</b> — {probs[idx]:.4f}</li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# FOOTER
# ==============================
st.markdown(
    "<div style='text-align:center; color:#6b7280; margin-top:16px;'>Model: CNN MobileNetV2 • Output: Top-3 Prediksi</div>",
    unsafe_allow_html=True
)
