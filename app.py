import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Healthcare Multi Disease Predictor")
st.title("🏥 Multi-Disease Deep Learning Predictor")

disease = st.selectbox(
    "Select Disease",
    ["Diabetes", "Heart Disease", "Kidney Disease", "Liver Disease"]
)

def load_resources(disease):

    if disease == "Diabetes":
        df = pd.read_csv("datasets/diabetes.csv")
        model = load_model("diabetes_model.h5")

    elif disease == "Heart Disease":
        df = pd.read_csv("datasets/heart.csv")
        model = load_model("heart_model.h5")

    elif disease == "Kidney Disease":
        df = pd.read_csv("datasets/kidney.csv")
        model = load_model("kidney_model.h5")

    else:
        df = pd.read_csv("datasets/liver.csv")
        model = load_model("liver_model.h5")

    df = df.fillna(df.mean())
    X = df.iloc[:, :-1]

    scaler = StandardScaler()
    scaler.fit(X)

    return model, scaler, X.shape[1]

model, scaler, feature_count = load_resources(disease)

st.subheader("Enter Patient Details")

input_data = []

for i in range(feature_count):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    input_data.append(val)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    if prediction[0][0] > 0.5:
        st.error(f"{disease} Detected ⚠")
    else:
        st.success(f"No {disease} Detected ✅")
