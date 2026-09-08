import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Détecteur d'images IA")
st.write("Upload une photo et l'IA te dit si elle est réelle ou générée.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ai_detector_model.keras")

model = load_model()

uploaded_file = st.file_uploader("Choisis une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image envoyée", use_column_width=True)

    img_resized = image.resize((96, 96))
    img_array = np.expand_dims(np.array(img_resized), axis=0)
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success(f"✅ Probablement RÉELLE ({prediction*100:.1f}%)")
    else:
        st.error(f"🤖 Probablement générée par IA ({(1-prediction)*100:.1f}%)")
