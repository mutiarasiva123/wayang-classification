import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Klasifikasi Tokoh Wayang", layout="centered")

# === Sesuaikan urutan kelas sesuai training kamu ===
CLASS_NAMES = [
    'arjuna', 'bagong', 'bathara surya', 'bathara wisnu', 'gareng',
    'nakula', 'petruk', 'sadewa', 'semar', 'werkudara', 'yudistira'
]

MODEL_PATH = "cnn_mobilenetv2_wayang_final.h5"
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    # compile=False biar cepat dan menghindari warning metric
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

st.title("Klasifikasi Tokoh Wayang")
st.write("Upload gambar wayang, lalu model akan memprediksi kelasnya.")

uploaded = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_container_width=True)

    # Preprocess (sesuai MobileNetV2)
    img_resized = img.resize(IMG_SIZE)
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx])

    st.subheader("Hasil Prediksi")
    st.write(f"**Prediksi:** {pred_label}")
    st.write(f"**Confidence:** {pred_conf:.4f}")

    top3 = np.argsort(probs)[::-1][:3]
    st.subheader("Top-3 Prediksi")
    for i, idx in enumerate(top3, start=1):
        st.write(f"{i}. {CLASS_NAMES[idx]} — {probs[idx]:.4f}")

    st.subheader("Validasi (opsional)")
    true_label = st.selectbox("Pilih label asli jika kamu tahu:", ["(tidak tahu)"] + CLASS_NAMES)
    if true_label != "(tidak tahu)":
        if true_label == pred_label:
            st.success("✅ BENAR")
        else:
            st.error("❌ SALAH")
