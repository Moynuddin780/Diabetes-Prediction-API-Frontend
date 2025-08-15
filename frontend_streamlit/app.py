import streamlit as st
import requests
import os

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")
st.title("🩺 Diabetes Prediction (Pima)")

# Sidebar for API config
default_api = os.environ.get("API_URL", "http://127.0.0.1:8000")
api_url = st.sidebar.text_input("Backend API URL", value=default_api)

with st.form("patient_form"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=3)
        Glucose = st.number_input("Glucose", min_value=0.0, value=145.0)
    with col2:
        BloodPressure = st.number_input("BloodPressure", min_value=0.0, value=70.0)
        SkinThickness = st.number_input("SkinThickness", min_value=0.0, value=20.0)
    with col3:
        Insulin = st.number_input("Insulin", min_value=0.0, value=85.0)
        BMI = st.number_input("BMI", min_value=0.0, value=33.6)
    with col4:
        DiabetesPedigreeFunction = st.number_input("DPF", min_value=0.0, value=0.35, step=0.01, format="%.2f")
        Age = st.number_input("Age", min_value=0, step=1, value=29)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "Pregnancies": int(Pregnancies),
        "Glucose": float(Glucose),
        "BloodPressure": float(BloodPressure),
        "SkinThickness": float(SkinThickness),
        "Insulin": float(Insulin),
        "BMI": float(BMI),
        "DiabetesPedigreeFunction": float(DiabetesPedigreeFunction),
        "Age": int(Age),
    }
    try:
        resp = requests.post(f"{api_url}/predict", json=payload, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Result: {data['result']}")
            st.write(f"Prediction: **{data['prediction']}** | Confidence: **{data['confidence']}**")
        else:
            st.error(f"API Error [{resp.status_code}]: {resp.text}")
    except Exception as e:
        st.error(f"Failed to contact API: {e}")

st.divider()
if st.button("Show Metrics"):
    try:
        r = requests.get(f"{api_url}/metrics", timeout=20)
        st.json(r.json())
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")