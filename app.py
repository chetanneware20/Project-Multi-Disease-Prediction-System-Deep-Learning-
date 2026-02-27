import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Healthcare Multi-Disease Predictor")

st.title("🏥 Healthcare Deep Learning Disease Prediction")

# Disease selection
disease = st.selectbox(
    "Select Disease to Predict",
    ["Diabetes", "Heart Disease", "Kidney Disease", "Liver Disease"]
)

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

def load_selected_model(disease):
    if disease == "Diabetes":
        return load_model("diabetes_model.h5")
    elif disease == "Heart Disease":
        return load_model("heart_model.h5")
    elif disease == "Kidney Disease":
        return load_model("kidney_model.h5")
    elif disease == "Liver Disease":
        return load_model("liver_model.h5")

model = load_selected_model(disease)

st.subheader("Enter Patient Details")

# Example generic inputs (modify based on dataset)
input_data = []

for i in range(8):  # Adjust number based on dataset features
    val = st.number_input(f"Feature {i+1}")
    input_data.append(val)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    if prediction[0][0] > 0.5:
        st.error(f"{disease} Detected ⚠")
    else:
        st.success(f"No {disease} Detected ✅")
