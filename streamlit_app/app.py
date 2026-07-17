

import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os

API_URL = os.getenv("CHURN_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Telco Churn Prediction", page_icon="📉", layout="wide")

st.title("📉 Telco Customer Churn Prediction")
st.caption("Müşteri bilgilerini gir, modelin churn (kayıp) olasılığını ve en etkili faktörleri gör.")

col_form, col_result = st.columns([1, 1])

with col_form:
    st.subheader("Müşteri Bilgileri")

    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Cinsiyet", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (ay)", 0, 72, 5)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with c2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 85.5)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 427.5)

    predict_btn = st.button("Churn Tahmini Yap", type="primary", use_container_width=True)

with col_result:
    st.subheader("Sonuç")

    if predict_btn:
        payload = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            prob = result["churn_probability"]
            risk = result["risk_level"]
            will_churn = result["churn_prediction"]

            color = {"Düşük": "green", "Orta": "orange", "Yüksek": "red"}[risk]

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                title={"text": f"Churn Olasılığı — Risk: {risk}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 30], "color": "#d4f7d4"},
                        {"range": [30, 60], "color": "#fff3cd"},
                        {"range": [60, 100], "color": "#f8d7da"},
                    ],
                },
            ))
            st.plotly_chart(fig, use_container_width=True)

            if will_churn:
                st.error(f"⚠️ Bu müşteri churn edebilir (eşik: {result['threshold_used']:.2f})")
            else:
                st.success(f"✅ Bu müşteri sadık kalma eğiliminde (eşik: {result['threshold_used']:.2f})")

            st.caption(f"Model: {result['model_name']}")

        except requests.exceptions.ConnectionError:
            st.error("FastAPI servisine bağlanılamadı. Önce şunu çalıştır:\n`uvicorn api.main:app --reload`")
        except Exception as e:
            st.error(f"Hata: {e}")
    else:
        st.info("Soldaki formu doldurunuz.")

st.divider()

st.subheader("📊 Model Genel Açıklanabilirlik (SHAP)")
st.caption("Bu grafikler tüm modelin hangi feature'lara genel olarak nasıl tepki verdiğini gösterir (offline üretilmiştir).")

img_col1, img_col2 = st.columns(2)
try:
    with img_col1:
        st.image("../reports/shap_feature_importance.png", caption="En Etkili Feature'lar")
    with img_col2:
        st.image("../reports/shap_summary_beeswarm.png", caption="Feature Değeri vs Churn Etkisi")
except Exception:
    st.warning("SHAP görselleri henüz üretilmedi. Önce `src/shap_analysis.py` çalıştırılmalı.")