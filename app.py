
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
model = pickle.load(open('models/best_rf_model.pkl', 'rb'))

st.title("🏦 Loan Approval Predictor")

# User inputs — Only current features used
income = st.number_input("Annual Income", min_value=0)
loan_amt = st.number_input("Loan Amount", min_value=0)
cibil = st.slider("CIBIL Score", min_value=300, max_value=900, value=700)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
asset_value = st.number_input("Asset Value", min_value=0)

# Map categorical inputs
education_map = {"Graduate": 1, "Not Graduate": 0}
self_emp_map = {"Yes": 1, "No": 0}

# Create DataFrame
input_data = pd.DataFrame({
    'income_annum': [income],
    'loan_amount': [loan_amt],
    'cibil_score': [cibil],
    'education': [education_map[education]],
    'self_employed': [self_emp_map[self_employed]],
    'asset_value': [asset_value]
})

if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)
    confidence = model.predict_proba(input_data)[0][int(prediction[0])] * 100
    if prediction[0] == 1:
        st.success(f"✅ Loan Approved with {confidence:.2f}% confidence.")
    else:
        st.error(f"❌ Loan Rejected with {confidence:.2f}% confidence.")
