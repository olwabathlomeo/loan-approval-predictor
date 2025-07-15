
import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("best_rf_model.pkl")

st.title("🏦 Loan Approval Prediction App")
st.markdown("Enter applicant details below to check if the loan is likely to be approved.")

# === INPUT FIELDS ===
no_of_dependents = st.number_input("👨‍👩‍👧 Number of Dependents", min_value=0, step=1)
education = st.selectbox("🎓 Education Level", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("💼 Self Employed?", ["No", "Yes"])
income_annum = st.number_input("📥 Annual Income", min_value=0)
loan_amount = st.number_input("💰 Loan Amount", min_value=0)
loan_term = st.number_input("📆 Loan Term (Years)", min_value=1)
cibil_score = st.slider("📊 CIBIL Score", 300, 900)
res_assets = st.number_input("🏠 Residential Assets Value", min_value=0)
com_assets = st.number_input("🏢 Commercial Assets Value", min_value=0)
lux_assets = st.number_input("🚗 Luxury Assets Value", min_value=0)
bank_assets = st.number_input("🏦 Bank Asset Value", min_value=0)

# === ENCODING ===
education = 0 if education == "Graduate" else 1
self_employed = 0 if self_employed == "No" else 1

# === PREDICTION ===
if st.button("🔍 Predict Loan Status"):
    input_data = pd.DataFrame([{
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": res_assets,
        "commercial_assets_value": com_assets,
        "luxury_assets_value": lux_assets,
        "bank_asset_value": bank_assets
    }])

    prediction = model.predict(input_data)[0]
    status = "✅ Approved" if prediction == 0 else "❌ Rejected"
    st.success(f"Loan Status: {status}")
