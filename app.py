import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(page_title="BMW Price Predictor", page_icon="🚗", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "best_model.pkl")

@st.cache_data
def load_metadata():
    with open(MODEL_DIR / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
metadata = load_metadata()
options = metadata["feature_options"]
ranges = metadata["numeric_ranges"]

st.title("🚗 BMW Used Car Price Prediction")
st.write("กรอกข้อมูลรถ BMW แล้วให้โมเดลช่วยประเมินราคามือสอง")

with st.expander("ดูข้อมูลโมเดล"):
    st.write(f"Best Model: {metadata['best_model_name']}")
    st.write(f"Test MAE: {metadata['best_model_test_mae']:.2f}")
    st.write(f"Test RMSE: {metadata['best_model_test_rmse']:.2f}")
    st.write(f"Test R²: {metadata['best_model_test_r2']:.4f}")

with st.form("predict_form"):
    model_name = st.selectbox("Model", options["model"])
    transmission = st.selectbox("Transmission", options["transmission"])
    fuel_type = st.selectbox("Fuel Type", options["fuelType"])

    year = st.number_input("Year", min_value=int(ranges["year"]["min"]), max_value=2026, value=int(ranges["year"]["median"]), step=1)
    mileage = st.number_input("Mileage", min_value=0, value=int(ranges["mileage"]["median"]), step=100)
    tax = st.number_input("Tax", min_value=0, value=int(ranges["tax"]["median"]), step=5)
    mpg = st.number_input("MPG", min_value=0.1, value=float(ranges["mpg"]["median"]), step=0.1)
    engine_size = st.number_input("Engine Size", min_value=0.1, value=float(ranges["engineSize"]["median"]), step=0.1)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([
        {
            "model": model_name,
            "year": year,
            "transmission": transmission,
            "mileage": mileage,
            "fuelType": fuel_type,
            "tax": tax,
            "mpg": mpg,
            "engineSize": engine_size,
        }
    ])
    pred = float(model.predict(input_df)[0])
    rmse = float(metadata["best_model_test_rmse"])
    st.success(f"Predicted Price: £{pred:,.2f}")
    st.info(f"Estimated Range: £{max(0, pred-rmse):,.2f} - £{pred+rmse:,.2f}")
    st.dataframe(input_df, use_container_width=True)
    st.caption("ผลลัพธ์นี้ใช้เพื่อการศึกษาและการประมาณเบื้องต้นเท่านั้น")
