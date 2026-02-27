import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.set_page_config(page_title="Healthcare Multi Disease Predictor")
st.title("🏥 Multi-Disease Deep Learning Predictor")

disease = st.selectbox(
    "Select Disease",
    ["Diabetes", "Heart Disease", "Kidney Disease", "Liver Disease"]
)

@st.cache_resource
def train_model(disease):

    if disease == "Diabetes":
        df = pd.read_csv("datasets/diabetes.csv")
    elif disease == "Heart Disease":
        df = pd.read_csv("datasets/heart.csv")
    elif disease == "Kidney Disease":
        df = pd.read_csv("datasets/kidney.csv")
    else:
        df = pd.read_csv("datasets/liver.csv")

    df = df.fillna(df.mean())

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X.shape[1]))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=30, batch_size=4, verbose=0)

    return model, scaler, X.shape[1]

model, scaler, feature_count = train_model(disease)

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
