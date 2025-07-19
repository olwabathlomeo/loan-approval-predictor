import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.markdown("This app predicts whether a loan will be **Approved** or **Rejected** based on applicant details. It also provides a SHAP explanation showing which features influenced the decision.")

# Load trained model
try:
    with open('best_rf_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Collect user input
st.subheader("📋 Applicant Details")

no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income (in ₹)", min_value=0)
loan_amount = st.number_input("Loan Amount (in ₹)", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=1)
cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=700)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# Prepare data for prediction
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

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0]

    if prediction[0] == 1:
        st.success(f"✅ Loan Approved with confidence of {round(probability[1]*100, 1)}%")
    else:
        st.error(f"❌ Loan Rejected with confidence of {round(probability[0]*100, 1)}%")

        # SHAP Explanation
    st.subheader("🔍 Explanation (SHAP)")
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        # Select class index for binary classification
        class_index = 1 if isinstance(shap_values, list) else 0

        # Only plot for a single sample
        shap.initjs()
        st_shap = st.components.v1.html(
            shap.force_plot(
                base_value=explainer.expected_value[class_index],
                shap_values=shap_values[class_index][0],
                features=input_df.iloc[0],
                feature_names=input_df.columns.tolist(),
                matplotlib=False
            ).html(),
            height=300,
        )
    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")
