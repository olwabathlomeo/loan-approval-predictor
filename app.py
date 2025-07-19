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
st.markdown("This app predicts whether a loan will be **Approved** or **Rejected** and explains the decision using SHAP.")

# Load trained model
try:
    with open('best_rf_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Collect input
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

# Prepare input
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

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0]

    if prediction[0] == 1:
        st.success(f"✅ Loan Approved with {round(probability[1]*100, 1)}% confidence.")
    else:
        st.error(f"❌ Loan Rejected with {round(probability[0]*100, 1)}% confidence.")

    # SHAP Explanation
    st.subheader("🔍 Explanation (SHAP)")

    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        # Handle binary classification case
        class_idx = 1 if isinstance(shap_values, list) and len(shap_values) > 1 else 0
        shap.initjs()

        # Generate and display force plot (SHAP v0.20+ compatible)
        force_plot = shap.plots.force(
            explainer.expected_value[class_idx],
            shap_values[class_idx][0] if isinstance(shap_values, list) else shap_values[0],
            input_df.iloc[0],
            matplotlib=False,
            show=False
        )

        components.html(force_plot.html(), height=300)
    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")
