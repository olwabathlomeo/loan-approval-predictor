import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Load model
with open('best_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# App UI
st.title("🏦 Loan Approval Predictor")
# ... input fields ...

# Create input_df from user input
# input_df = pd.DataFrame({...})

if st.button("Predict"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0]

    if prediction[0] == 1:
        st.success(f"✅ Loan Approved with confidence of {round(probability[1] * 100, 1)}%")
    else:
        st.error(f"❌ Loan Rejected with confidence of {round(probability[0] * 100, 1)}%")

    # 🔍 SHAP Explanation
    st.subheader("🔍 Explanation (SHAP)")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        class_index = 1 if isinstance(shap_values, list) else 0

        fig, ax = plt.subplots(figsize=(10, 3))
        shap.force_plot(
            base_value=explainer.expected_value[class_index] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
            shap_values=shap_values[class_index][0] if isinstance(shap_values, list) else shap_values[0],
            features=input_df.iloc[0],
            matplotlib=True,
            show=False
        )
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")
