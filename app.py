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
income_annum = st.sidebar.number_input("Annual Income (KSh)", min_value=10000, max_value=1000000, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount (KSh)", min_value=5000, max_value=500000, step=1000)
cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 650)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
asset_value = st.sidebar.number_input("Asset Value (KSh)", min_value=0, max_value=1000000, step=5000)

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
