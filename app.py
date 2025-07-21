import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Page setup
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

# File check and load
model_path = "best_rf_model.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error(f"🚫 Model or scaler file missing! Files in directory: {os.listdir('.')}")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.title("🏦 Loan Approval Predictor (Random Forest)")
st.markdown("Enter applicant details below to predict loan approval status:")

# Input form
dependents = st.number_input("No. of Dependents", min_value=0, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income = st.number_input("Annual Income (KES)", min_value=0)
loan_amount = st.number_input("Loan Amount (KES)", min_value=0)
loan_term = st.number_input("Loan Term (Months)", min_value=0)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
res_assets = st.number_input("Residential Assets Value", min_value=0)
com_assets = st.number_input("Commercial Assets Value", min_value=0)
lux_assets = st.number_input("Luxury Assets Value", min_value=0)
bank_assets = st.number_input("Bank Asset Value", min_value=0)

# Prepare input data
input_data = {
    "no_of_dependents": dependents,
    "education": 1 if education == "Graduate" else 0,
    "self_employed": 1 if self_employed == "Yes" else 0,
    "income_annum": income,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "residential_assets_value": res_assets,
    "commercial_assets_value": com_assets,
    "luxury_assets_value": lux_assets,
    "bank_asset_value": bank_assets
}

input_df = pd.DataFrame([input_data])

# Predict
if st.button("🔍 Predict Loan Status"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    confidence = model.predict_proba(scaled_input)[0][prediction]

    if prediction == 1:
        st.success(f"✅ Loan Approved (Confidence: {confidence:.2%})")
    else:
        st.error(f"❌ Loan Rejected (Confidence: {confidence:.2%})")

    # SHAP Explanation
    st.subheader("📊 SHAP Feature Contribution")
    explainer = shap.Explainer(model, input_df)
    shap_values = explainer(input_df)

    shap_df = pd.DataFrame({
        "Feature": input_df.columns,
        "SHAP Value": shap_values.values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=True)

    colors = shap_df["SHAP Value"].apply(lambda x: "green" if x > 0 else "red")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
    ax.set_xlabel("SHAP Value (Impact on Decision)")
    ax.set_title("Top Feature Contributions")
    st.pyplot(fig)

    # Adaptive explanation
    st.markdown("#### 🧠 Interpretation:")
    if prediction == 1:
        st.markdown("""
        - ✅ **Green bars**: Features that **contributed positively** to loan approval.
        - ❌ **Red bars**: Features that could have reduced confidence but were outweighed.
        """)
    else:
        st.markdown("""
        - ❌ **Red bars**: Features that **led to rejection** of the loan.
        - ✅ **Green bars**: Positive features, but not strong enough to get approval.
        """)

