import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")

# Load model and explainer
with open('best_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('shap_explainer.pkl', 'rb') as explainer_file:
    explainer = pickle.load(explainer_file)

# Input fields for all required features
income_annum = st.number_input("📥 Annual Income", min_value=0.0, value=50000.0)
loan_amount = st.number_input("💰 Loan Amount", min_value=0.0, value=10000.0)
cibil_score = st.slider("📊 CIBIL Score", 300, 900, value=700)
education = st.selectbox("🎓 Education Level", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("💼 Self Employed?", ["Yes", "No"])

bank_asset_value = st.number_input("🏦 Bank Asset Value", min_value=0.0, value=15000.0)
commercial_assets_value = st.number_input("🏭 Commercial Assets Value", min_value=0.0, value=20000.0)
luxury_assets_value = st.number_input("💎 Luxury Assets Value", min_value=0.0, value=5000.0)
residential_assets_value = st.number_input("🏠 Residential Assets Value", min_value=0.0, value=25000.0)
loan_term = st.number_input("📅 Loan Term (months)", min_value=1, value=36)
no_of_dependents = st.number_input("👨‍👩‍👧 Number of Dependents", min_value=0, value=0)

education_encoded = 1 if education == "Graduate" else 0
self_employed_encoded = 1 if self_employed == "Yes" else 0

# Update the DataFrame to include the new feature, in the order your model was trained on!
input_data = pd.DataFrame([[
    income_annum,
    loan_amount,
    cibil_score,
    education_encoded,
    self_employed_encoded,
    bank_asset_value,
    commercial_assets_value,
    luxury_assets_value,
    residential_assets_value,
    loan_term,
    no_of_dependents
]], columns=[
    "income_annum",
    "loan_amount",
    "cibil_score",
    "education",
    "self_employed",
    "bank_asset_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "residential_assets_value",
    "loan_term",
    "no_of_dependents"
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
        shap_values = explainer.shap_values(input_data)
        import shap
        if isinstance(shap_values, list):
            values = shap_values[1][0]
            base_value = explainer.expected_value[1]
        else:
            values = shap_values[0]
            base_value = explainer.expected_value
        shap.waterfall_plot(shap.Explanation(
            values=values,
            base_values=base_value,
            data=input_data.iloc[0],
            feature_names=input_data.columns.tolist()
        ), max_display=10)
        st.pyplot()
    except Exception as e:
        st.error(f"Prediction or explanation failed: {e}")
