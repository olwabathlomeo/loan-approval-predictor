import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")

with open('best_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('shap_explainer.pkl', 'rb') as explainer_file:
    explainer = pickle.load(explainer_file)

no_of_dependents = st.number_input("👨‍👩‍👧 Number of Dependents", min_value=0, value=0)
education = st.selectbox("🎓 Education Level", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("💼 Self Employed?", ["Yes", "No"])
income_annum = st.number_input("📥 Annual Income", min_value=0.0, value=50000.0)
loan_amount = st.number_input("💰 Loan Amount", min_value=0.0, value=10000.0)
loan_term = st.number_input("📅 Loan Term (months)", min_value=1, value=36)
cibil_score = st.slider("📊 CIBIL Score", 300, 900, value=700)
residential_assets_value = st.number_input("🏠 Residential Assets Value", min_value=0.0, value=25000.0)
commercial_assets_value = st.number_input("🏭 Commercial Assets Value", min_value=0.0, value=20000.0)
luxury_assets_value = st.number_input("💎 Luxury Assets Value", min_value=0.0, value=5000.0)
bank_asset_value = st.number_input("🏦 Bank Asset Value", min_value=0.0, value=15000.0)

education_encoded = 1 if education == "Graduate" else 0
self_employed_encoded = 1 if self_employed == "Yes" else 0

input_data = pd.DataFrame([[
    no_of_dependents,
    education_encoded,
    self_employed_encoded,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
]], columns=[
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
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
        pred_class = int(prediction)
        # --- FIX: Always extract the single explanation as a 1D array ---
        if isinstance(shap_values, list):
            # Each item in shap_values is (n_samples, n_features), so [sample][feature]
            # For a single sample, shap_values[pred_class][0] is shape (n_features,)
            values = shap_values[pred_class][0]
            base_value = explainer.expected_value[pred_class]
        elif shap_values.ndim == 3:
            # shape (n_samples, n_classes, n_features)
            values = shap_values[0, pred_class]
            base_value = explainer.expected_value[pred_class]
        elif shap_values.ndim == 2 and shap_values.shape[0] == 1:
            # shape (1, n_features)
            values = shap_values[0]
            base_value = explainer.expected_value
        else:
            raise ValueError("Unexpected shape for shap_values: {}".format(shap_values.shape))

        fig, ax = plt.subplots()
        shap.waterfall_plot(
            shap.Explanation(
                values=values,
                base_values=base_value,
                data=input_data.iloc[0],
                feature_names=input_data.columns.tolist()
            ),
            max_display=10,
            show=False
        )
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Prediction or explanation failed: {e}")
