import os

import altair as alt
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FEATURE_LABELS = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "ph": "Soil pH",
    "rainfall": "Rainfall (mm)",
}

st.set_page_config(page_title="Smart Crop Recommendation", page_icon="🌱", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("🌱 Smart Crop Recommendation System")
st.write(
    "Recommends the most suitable crop based on soil nutrients and weather conditions, "
    "using a Random Forest classifier trained on 2,200 samples of 22 crops."
)

left, right = st.columns([1, 1.4], gap="large")

with left:
    st.subheader("Soil & weather parameters")
    N = st.number_input(FEATURE_LABELS["N"], min_value=0, max_value=200, value=90)
    P = st.number_input(FEATURE_LABELS["P"], min_value=0, max_value=200, value=42)
    K = st.number_input(FEATURE_LABELS["K"], min_value=0, max_value=250, value=43)
    temperature = st.number_input(FEATURE_LABELS["temperature"], min_value=0.0, max_value=50.0, value=20.9, step=0.1)
    humidity = st.number_input(FEATURE_LABELS["humidity"], min_value=0.0, max_value=100.0, value=82.0, step=0.1)
    ph = st.number_input(FEATURE_LABELS["ph"], min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.number_input(FEATURE_LABELS["rainfall"], min_value=0.0, max_value=400.0, value=202.9, step=0.1)
    predict = st.button("Predict crop", type="primary", width="stretch")

with right:
    st.subheader("Prediction result")
    if predict:
        sample = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES)
        probabilities = model.predict_proba(sample)[0]
        ranking = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
        best_crop, best_prob = ranking[0]

        st.success(f"Recommended crop: **{best_crop.capitalize()}**")
        st.metric("Model confidence", f"{best_prob:.0%}")
        st.progress(float(best_prob))

        st.markdown("**Top 3 suggestions**")
        st.table(
            pd.DataFrame(
                {"Crop": [c.capitalize() for c, _ in ranking[:3]], "Probability": [f"{p:.0%}" for _, p in ranking[:3]]}
            )
        )
    else:
        st.info("Enter the soil and weather values and press **Predict crop**.")

    st.markdown("**Most important features of the model**")
    importance = pd.DataFrame(
        {"Feature": [FEATURE_LABELS[f] for f in FEATURES], "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)
    chart = (
        alt.Chart(importance)
        .mark_bar(color="#2e9e5b")
        .encode(
            x=alt.X("Importance:Q", title="Importance", axis=alt.Axis(format="%")),
            y=alt.Y("Feature:N", sort="-x", title=None),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".1%")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")
    st.caption("Feature importance of the Random Forest model (higher = more influence on the predictions overall).")
