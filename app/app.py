import streamlit as st
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")

model = joblib.load(model_path)
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌱"
)

st.title("Smart Crop Recommendation System 🌱")

st.write("""
This app recommends the best crop based on soil nutrients and weather conditions.
""")

N = st.number_input("Nitrogen (N)",min_value=0)
P = st.number_input("Phosphorus (P)", min_value=0)
K = st.number_input("Potassium (K)", min_value=0)
temperature = st.number_input("Temperature", min_value=0.0)
humidity = st.number_input("Humidity", min_value=0.0)
ph = st.number_input("pH", min_value=0.0)
rainfall = st.number_input("Rainfall", min_value=0.0)

if st.button("Predict Corp"):
    sample= [[N,P,K,temperature,humidity,ph,rainfall]]
    prediction = model.predict(sample)
    st.success(f"Recommand Corp :{prediction[0]}")
    