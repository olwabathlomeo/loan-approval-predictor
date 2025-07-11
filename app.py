
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Try loading the saved model
try:
    model = pickle.load(open('best_rf_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("❌ Model file not found. Please make sure 'best_rf_model.pkl' is in the same folder as this app.py file.")
    st.stop()

# App title and description
st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details below to check if the loan is likely to be approved.")

# Collecting user inputs
income = st.number_input("📥 Annual Income", min_value=0, format="%d")
loan_amount = st.number_input("💰 Loan Amount", min_value=0, format="%d")
cibil_score = st.slider("📊 CIBIL Score", min_value=300, max_value=900, step=1)
education = st.selectbox("🎓 Education Level", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("💼 Self Employed?", ["No", "Yes"])

# Encode the categorical values just like in training
education_encoded = 0 if education == "Graduate" else 1
self_employed_encoded = 0 if self_employed == "No" else 1

# Prepare input data for prediction
features = np.array([[income, loan_amount, cibil_score, education_encoded, self_employed_encoded]])

# Prediction
if st.button("🔍 Predict Loan Approval"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][prediction]

    if prediction == 0:
        st.success(f"✅ Loan Approved with confidence {prob:.2%}")
    else:
        st.error(f"❌ Loan Rejected with confidence {prob:.2%}")

# Footer
st.markdown("---")
st.caption("Built using Streamlit | ML Model: Random Forest Classifier")
