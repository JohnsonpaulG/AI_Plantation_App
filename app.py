import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import datetime

st.set_page_config(page_title="🌿 AI Plantation App", layout="centered")

st.title("🌿 AI-Powered Plantation Management App")
st.caption("Developed by Students | Guided by Faculty")

tab1, tab2 = st.tabs(["Plant Health Detection", "Watering Time Prediction"])

# ------------ PLANT HEALTH ------------
with tab1:
    st.header("🌱 Check Plant Health (AI Image Recognition)")
    cnn_model = tf.keras.models.load_model("plant_health_model.h5")

    img_file = st.file_uploader("Upload a leaf photo", type=["jpg", "png", "jpeg"])
    if img_file:
        img = Image.open(img_file).resize((128, 128))
        st.image(img, caption="Uploaded Leaf Image", use_container_width=True)
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)
        prediction = cnn_model.predict(arr)[0][0]
        status = "Healthy ✅" if prediction < 0.5 else "Unhealthy ⚠️"
        st.success(f"Plant Health: **{status}**")

# ------------ WATERING PREDICTION ------------
with tab2:
    st.header("💧 Predict Next Watering Time")
    model = joblib.load("watering_model.pkl")

    temp = st.slider("Temperature (°C)", 10, 45, 28)
    humidity = st.slider("Humidity (%)", 10, 90, 55)
    soil = st.slider("Soil Moisture (%)", 0, 100, 40)

    if st.button("Predict"):
        pred_hours = model.predict([[temp, humidity, soil]])[0]
        next_time = datetime.datetime.now() + datetime.timedelta(hours=pred_hours)
        st.info(f"Next watering in {round(pred_hours,1)} hours ({next_time.strftime('%I:%M %p')})")
        st.balloons()
