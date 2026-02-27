# train_model.py

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model

def create_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_save_model(dataset_path, model_name):
    df = pd.read_csv(dataset_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = create_model(X.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

    model.save(f"{model_name}.h5")

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

# Train models
train_and_save_model("datasets/diabetes.csv", "diabetes_model")
train_and_save_model("datasets/heart.csv", "heart_model")
train_and_save_model("datasets/kidney.csv", "kidney_model")
train_and_save_model("datasets/liver.csv", "liver_model")
