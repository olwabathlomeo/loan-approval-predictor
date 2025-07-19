import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Set page config
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.markdown("This app predicts whether a loan will be **Approved** or **Rejected** and provides a SHAP explanation.")

# Load model
try:
    with open("best_rf_model.pkl", "rb") as file:
        model = pickle.load(file)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input form
st.subheader("📋 Applicant Details")
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income (₹)", min_value=0)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
loan_term = st.number_input("Loan Term (months)", min_value=1)
cibil_score = st.slider("CIBIL Score", 300, 900, 700)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# Prepare input for prediction
input_data = {
    "no_of_dependents": no_of_dependents,
    "education": 1 if education == "Graduate" else 0,
    "self_employed": 1 if self_employed == "Yes" else 0,
    "income_annum": income_annum,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "residential_assets_value": residential_assets_value,
    "commercial_assets_value": commercial_assets_value,
    "luxury_assets_value": luxury_assets_value,
    "bank_asset_value": bank_asset_value
}

input_df = pd.DataFrame([input_data])

# Predict and explain
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.success(f"✅ Loan Approved with confidence of {round(prob[1]*100, 2)}%")
    else:
        st.error(f"❌ Loan Rejected with confidence of {round(prob[0]*100, 2)}%")

    # SHAP Explanation
    st.subheader("🔍 Explanation (SHAP)")
    try:
        explainer = shap.Explainer(model, input_df)
        shap_values = explainer(input_df)

        # Use the new SHAP plots.force() API
        force_plot_html = shap.plots.force(
            shap_values[0].base_values,
            shap_values[0].values,
            input_df.iloc[0],
            matplotlib=False,
            show=False
        ).html()

        # Display in Streamlit
        components.html(shap.getjs() + force_plot_html, height=300)
    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")
