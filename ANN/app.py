import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import load_model
from sklearn.datasets import load_breast_cancer

st.set_page_config(
    page_title="ANN Cancer Prediction",
    page_icon="🧠",
    layout="wide"
)

model = load_model("ann_model.h5")
scaler = joblib.load("scaler.pkl")

data = load_breast_cancer()

feature_names = data.feature_names

st.title("🧠 ANN Cancer Prediction")

st.write(
    "Artificial Neural Network trained on Breast Cancer Dataset."
)

inputs = []

col1, col2 = st.columns(2)

for i in range(15):
    value = col1.number_input(
        feature_names[i],
        value=float(data.data[:, i].mean())
    )
    inputs.append(value)

for i in range(15, 30):
    value = col2.number_input(
        feature_names[i],
        value=float(data.data[:, i].mean())
    )
    inputs.append(value)

if st.button("Predict"):

    arr = np.array(inputs).reshape(1, -1)

    arr = scaler.transform(arr)

    pred = model.predict(arr)[0][0]

    if pred > 0.5:
        st.success(
            f"Benign (Non-Cancerous)\n\nConfidence: {pred:.2%}"
        )
    else:
        st.error(
            f"Malignant (Cancerous)\n\nConfidence: {(1-pred):.2%}"
        )