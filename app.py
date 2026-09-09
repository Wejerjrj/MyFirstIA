import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib
from PIL import Image

st.title("Détecteur d'images IA")
st.write("Upload une photo et l'IA te dit si elle est réelle ou générée.")

@st.cache_resource
def load_model():
    m = tf.keras.models.load_model("ai_detector_model.keras")
    # --- FIX KERAS ---
    # On force la construction du graphe symbolique pour que model.output soit défini
    _ = m(tf.keras.Input(shape=(96, 96, 3)))
    return m

model = load_model()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):
    base_model = model.layers[1]
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if predictions[0][0] > 0.5:
            loss = predictions[:, 0]
        else:
            loss = -predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(image, heatmap, alpha=0.4):
    jet = matplotlib.colormaps["jet"]
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255)).resize(image.size)

    overlaid = np.array(jet_heatmap) * alpha + np.array(image) * (1 - alpha)
    return Image.fromarray(np.uint8(overlaid))

uploaded_file = st.file_uploader("Choisis une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # --- FIX STREAMLIT ---
    st.image(image, caption="Image envoyée", width="stretch")

    img_resized = image.resize((96, 96))
    img_array = np.expand_dims(np.array(img_resized), axis=0).astype("float32") / 255.0
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success(f"✅ Probablement RÉELLE ({prediction*100:.1f}%)")
    else:
        st.error(f"🤖 Probablement générée par IA ({(1-prediction)*100:.1f}%)")

    heatmap = make_gradcam_heatmap(img_array, model)
    overlay = overlay_heatmap(image, heatmap)
    
    # --- FIX STREAMLIT ---
    st.image(overlay, caption="Zones ayant influencé la décision", width="stretch")
