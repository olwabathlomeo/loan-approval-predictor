import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt

# Load the trained model
with open('best_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# App config
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.markdown("This app predicts whether a loan will be **Approved** or **Rejected**, and explains the decision using SHAP.")

st.sidebar.header("📋 Enter Applicant Information")

# Input fields (11 features)
no_of_dependents = st.sidebar.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.sidebar.number_input("Annual Income", min_value=0, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=1000)
loan_term = st.sidebar.number_input("Loan Term (months)", min_value=1, step=1)
cibil_score = st.sidebar.slider("CIBIL Score", min_value=300, max_value=900, value=650)
residential_assets_value = st.sidebar.number_input("Residential Assets Value", min_value=0, step=1000)
commercial_assets_value = st.sidebar.number_input("Commercial Assets Value", min_value=0, step=1000)
luxury_assets_value = st.sidebar.number_input("Luxury Assets Value", min_value=0, step=1000)
bank_asset_value = st.sidebar.number_input("Bank Asset Value", min_value=0, step=1000)

# Encoding categorical values
education_encoded = 1 if education == "Graduate" else 0
self_employed_encoded = 1 if self_employed == "Yes" else 0

# Create DataFrame from inputs
input_data = pd.DataFrame([{
    'no_of_dependents': no_of_dependents,
    'education': education_encoded,
    'self_employed': self_employed_encoded,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets_value,
    'commercial_assets_value': commercial_assets_value,
    'luxury_assets_value': luxury_assets_value,
    'bank_asset_value': bank_asset_value
}])

# Prediction
if st.button("Predict Loan Status"):
    prediction = model.predict(i
