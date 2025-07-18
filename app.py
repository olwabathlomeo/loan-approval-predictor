import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.markdown("This app predicts whether a loan will be **Approved** or **Rejected** based on applicant data.")

# Set matplotlib to not show global pyplot deprecation warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load trained model
if not os.path.exists('best_rf_model.pkl') or not os.path.exists('shap_explainer.pkl'):
    st.error("Model or SHAP explainer file not found. Please check your deployment.")
    st.stop()

with open('best_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('shap_explainer.pkl', 'rb') as explainer_file:
    explainer = pickle.load(explainer_file)

# Input fields
income_annum = st.number_input("📥 Annual Income", min_value=0.0, value=50000.0)
loan_amount = st.number_input("💰 Loan Amount", min_value=0.0, value=10000.0)
cibil_score = st.slider("📊 CIBIL Score", 300, 900, value=700)
education = st.selectbox("🎓 Education Level", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("💼 Self Employed?", ["Yes", "No"])
asset_value = st.number_input("💎 Total Asset Value", min_value=0.0, value=20000.0)

# Encode inputs
education_encoded = 1 if education == "Graduate" else 0
self_employed_encoded = 1 if self_employed == "Yes" else 0

# Prepare input DataFrame
input_data = pd.DataFrame([[
    income_annum,
    loan_amount,
    cibil_score,
    education_encoded,
    self_employed_encoded,
    asset_value
]], columns=[
    "income_annum",
    "loan_amount",
    "cibil_score",
    "education",
    "self_employed",
    "asset_value"
])

if st.button("🔍 Predict Loan Status"):
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][int(prediction)] * 100

        if prediction == 1:
            st.success(f"✅ Loan Approved with {proba:.2f}% confidence.")
        else:
            st.error(f"❌ Loan Rejected with {proba:.2f}% confidence.")

        st.markdown("### 🧠 SHAP Feature Impact")

        # Get SHAP values
        shap_values = explainer.shap_values(input_data)

        # For binary classifier (usually a list: [class0, class1])
        if isinstance(shap_values, list):
            values = shap_values[1][0]
            base_value = explainer.expected_value[1]
        else:
            values = shap_values[0]
            base_value = explainer.expected_value

        # Plot SHAP waterfall using legacy API for matplotlib compatibility
        shap.waterfall_plot(shap.Explanation(values=values,
                                             base_values=base_value,
                                             data=input_data.iloc[0],
                                             feature_names=input_data.columns.tolist()),
                            max_display=6)
        st.pyplot()
    except Exception as e:
        st.error(f"Prediction or explanation failed: {e}")
