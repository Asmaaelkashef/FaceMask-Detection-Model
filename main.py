import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import cv2

st.set_page_config(page_title="Mask Detection CNN", layout="centered")

model = load_model("face_mask_model.h5")

page_bg_color = """
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #000000D8; 
    }

    h1 {
        text-align: center;
        color: white;
        font-weight: bold;
    }

    .made-by {
        text-align: center;
        font-size: 23px;
        color: white;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    .emoji {
        font-size: 50px;
        color: yellow; 
        display: inline;
        vertical-align: middle;
        margin-right: 10px;
    }

</style>
"""
st.markdown(page_bg_color, unsafe_allow_html=True)

st.markdown('<div style="text-align: center;"><span class="emoji">😷</span><h1>Mask Detection Model</h1></div>', unsafe_allow_html=True)

st.markdown('<div class="made-by">Made with 💖 by Asmaa Elkashef</div>', unsafe_allow_html=True)


uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة المرفوعة", use_column_width=True)

    img_array = np.array(image.resize((128, 128))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "❌ No Mask" if prediction >= 0.5 else "😷 Mask"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")
