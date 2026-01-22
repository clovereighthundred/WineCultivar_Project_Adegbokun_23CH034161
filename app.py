import os
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

MODEL_PATH = os.path.join("model", "wine_cultivar_model.pkl")
FEATURES = ["alcohol", "malic_acid", "alcalinity_of_ash", "flavanoids", "color_intensity", "proline"]

st.set_page_config(page_title="Wine Cultivar Predictor", layout="centered")
st.title("Wine Cultivar Origin Prediction")

if not os.path.exists(MODEL_PATH):
    st.error("Saved model not found. Run the training script first to create model/wine_cultivar_model.pkl")
else:
    model = load(MODEL_PATH)

    st.subheader("Enter wine chemical properties (selected features)")

    # Provide sensible defaults (based on typical wine dataset ranges)
    alcohol = st.number_input("Alcohol", value=13.0, format="%.3f")
    malic_acid = st.number_input("Malic acid", value=2.0, format="%.3f")
    alcalinity_of_ash = st.number_input("Alcalinity of ash", value=17.0, format="%.3f")
    flavanoids = st.number_input("Flavanoids", value=2.0, format="%.3f")
    color_intensity = st.number_input("Color intensity", value=5.0, format="%.3f")
    proline = st.number_input("Proline", value=750.0, format="%.1f")

    input_df = pd.DataFrame([[
        alcohol, malic_acid, alcalinity_of_ash, flavanoids, color_intensity, proline
    ]], columns=FEATURES)

    if st.button("Predict cultivar"):
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None

        # Convert 0/1/2 to Cultivar 1/2/3 for display
        st.success(f"Predicted cultivar: Cultivar {int(pred) + 1}")

        if proba is not None:
            st.write("Class probabilities:")
            proba_df = pd.DataFrame({
                "Cultivar": [f"Cultivar {i+1}" for i in range(len(proba))],
                "Probability": proba
            })
            st.dataframe(proba_df)
        with st.expander("Show input features"):
            st.write(input_df)