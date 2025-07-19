import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")

st.title("🏦 Loan Approval Predictor")
st.markdown("""
This app predicts whether a loan will be **Approved** or **Rejected**, based on applicant details.
It also provides a SHAP explanation showing which features influenced the decision.
""")

# Load trained model
with open("best_rf_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load training data for SHAP explainer
X_train = pd.read_csv("X_train.csv")

# Sidebar - Input fields
st.sidebar.header("📝 Applicant Information")

no_of_dependents = st.sidebar.number_input("Number of Dependents")
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.sidebar.number_input("Annual Income (KSh)")
loan_amount = st.sidebar.number_input("Loan Amount (KSh)")
loan_term = st.sidebar.number_input("Loan Term (Months)")
cibil_score = st.sidebar.slider("CIBIL Score", min_value=300, max_value=900)  # Set actual min and max
residential_assets_value = st.sidebar.number_input("Residential Assets Value (KSh)")
commercial_assets_value = st.sidebar.number_input("Commercial Assets Value (KSh)")
luxury_assets_value = st.sidebar.number_input("Luxury Assets Value (KSh)")
bank_asset_value = st.sidebar.number_input("Bank Asset Value (KSh)")

# Encoding mappings
education_map = {"Graduate": 0, "Not Graduate": 1}
self_employed_map = {"No": 0, "Yes": 1}

# Prepare input using the correct column order
input_data = pd.DataFrame([[
    no_of_dependents,
    education_map[education],
    self_employed_map[self_employed],
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
]], columns=X_train.columns)


# Encoding mappings
education_map = {"Graduate": 0, "Not Graduate": 1}
self_employed_map = {"No": 0, "Yes": 1}

# Prepare input using the exact training columns
input_data = pd.DataFrame([[
    no_of_dependents,
    education_map[education],
    self_employed_map[self_employed],
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
]], columns=X_train.columns)


# Encoding mappings
education_map = {"Graduate": 0, "Not Graduate": 1}
self_employed_map = {"No": 0, "Yes": 1}

# Prepare input
input_data = pd.DataFrame([{
    "income_annum": income_annum,
    "loan_amount": loan_amount,
    "cibil_score": cibil_score,
    "education": education_map[education],
    "self_employed": self_employed_map[self_employed],
    "asset_value": asset_value
}])

# Make prediction
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    result = "✅ Loan Approved" if prediction == 0 else "❌ Loan Rejected"
    st.subheader("📌 Prediction Result")
    st.markdown(f"**{result}** with confidence of **{round(np.max(prediction_proba)*100, 2)}%**")

    # SHAP Explanation
    st.subheader("🔍 Explanation (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Plot SHAP summary bar chart
    shap.initjs()
    plt.figure()
    shap.summary_plot(shap_values[1], X_train, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')
