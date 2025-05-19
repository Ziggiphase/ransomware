# This Streamlit app trains a real hybrid CNN+RNN model on entropy dataset,
# allows inference from user input, and supports saving/reloading the model.

import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras import layers, models
import joblib

st.title("Ransomware Family Classifier")

# File upload or use existing CSV
data_source = st.radio("Select Data Source", ("Upload CSV", "Use Demo Dataset"))

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your entropy dataset (CSV format)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
else:
    demo_path = "formatted_entropy_dataset.csv"
    if os.path.exists(demo_path):
        df = pd.read_csv(demo_path)
    else:
        st.warning("Demo dataset not found.")
        st.stop()

if df is not None:
    st.subheader("Raw Dataset Preview")
    st.dataframe(df.head())

    # Identify string columns
    string_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Encode label (assume target column is 'family')
    if 'family' not in df.columns:
        st.error("Dataset must contain a 'family' column for classification.")
        st.stop()

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['family'])
    joblib.dump(label_encoder, "label_encoder.pkl")

    # Encode any other string columns (like file_type)
    df = pd.get_dummies(df, columns=[col for col in string_cols if col != 'family'])

    # Normalize features
    scaler = MinMaxScaler()
    feature_cols = [col for col in df.columns if col not in ['family', 'label']]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, "scaler.pkl")

    # Data split
    X = df[feature_cols].values.reshape(df.shape[0], df[feature_cols].shape[1], 1)  # sequence-like shape
    y = tf.keras.utils.to_categorical(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("Train Hybrid CNN+RNN Model")
    if st.button("Train Model"):
        model = models.Sequential([
            layers.Conv1D(32, kernel_size=2, activation='relu', input_shape=(X.shape[1], 1)),
            layers.MaxPooling1D(pool_size=2),
            layers.LSTM(32),
            layers.Dense(64, activation='relu'),
            layers.Dense(y.shape[1], activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(X_train, y_train, epochs=15, batch_size=16, validation_split=0.1)

        # Save model
        model.save("ransomware_model.h5")
        st.success("Model trained and saved as ransomware_model.h5")

        st.subheader("Evaluate on Test Set")
        loss, accuracy = model.evaluate(X_test, y_test)
        st.write(f"Test Accuracy: {accuracy * 100:.2f}%")

# Inference Section
st.subheader("Ransomware Family Inference")
model_file = "ransomware_model.h5"

if os.path.exists(model_file):
    model = tf.keras.models.load_model(model_file)
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")

    input_method = st.radio("Input method", ["Manual Entry", "Upload Entropy CSV"])

    if input_method == "Manual Entry":
        manual_vals = []
        for col in feature_cols:
            val = st.number_input(f"Enter value for {col}", value=0.5, step=0.01)
            manual_vals.append(val)

        if st.button("Predict Family"):
            manual_input = np.array(manual_vals).reshape(1, len(manual_vals))
            scaled = scaler.transform(manual_input).reshape(1, len(manual_vals), 1)
            prediction = model.predict(scaled)
            label = label_encoder.inverse_transform([np.argmax(prediction)])
            st.success(f"Predicted Ransomware Family: {label[0]}")

    else:
        predict_file = st.file_uploader("Upload a CSV with entropy feature values", type="csv")
        if predict_file:
            pred_df = pd.read_csv(predict_file)
            pred_df = pd.get_dummies(pred_df)

            for col in feature_cols:
                if col not in pred_df.columns:
                    pred_df[col] = 0
            pred_df = pred_df[feature_cols]  # re-order

            scaled_input = scaler.transform(pred_df.values).reshape(pred_df.shape[0], len(feature_cols), 1)
            predictions = model.predict(scaled_input)
            predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
            pred_df['Predicted_Family'] = predicted_labels
            st.write(pred_df[['Predicted_Family']])
else:
    st.warning("No trained model found. Please train the model first.")
