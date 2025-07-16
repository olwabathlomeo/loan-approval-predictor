readme_content = """# 🏦 Loan Approval Predictor - Machine Learning App

Predict whether a loan will be **Approved** or **Rejected** using applicant data like income, loan amount, CIBIL score, and employment status. Built with real-world data, trained using Random Forest, and deployed as a Streamlit web application.

---

## 📊 Project Overview

Loan evaluation is often slow, biased, or based on outdated methods. Our ML-powered app brings speed, accuracy, and fairness to the loan approval process.

- ✔️ Predictive model with over **86% accuracy**
- 💡 Trained on cleaned and encoded real-world loan data
- 🖥️ Streamlit interface for easy testing and visualization

---

## 📂 Project Structure


---

## 🧪 Features Used

| Feature         | Description                          |
|----------------|--------------------------------------|
| `income_annum` | Annual income of the applicant       |
| `loan_amount`  | Requested loan amount                |
| `cibil_score`  | Credit score of the applicant        |
| `education`    | Graduate / Not Graduate              |
| `self_employed`| Yes / No                             |

---

## 🤖 Model Performance

- **Model**: Random Forest Classifier  
- **Accuracy**: `86.3%`

**Evaluation Metrics:**
- Precision, Recall, F1-Score
- Confusion Matrix
- Feature Importance Visualization

---

## 🌐 App Demo (Streamlit)

The app allows users to input values and receive instant predictions.

**Example Inputs:**
- Annual Income: `550,000`
- Loan Amount: `150,000`
- CIBIL Score: `725`
- Education: `Graduate`
- Self-Employed: `No`

**Output:**  
✅ Loan Approved with **87.5% confidence**

---

## 📸 Visualizations

![CIBIL Score Boxplot](images/CIBIL_score_boxplot.png)  
![Feature Importance](images/feature_importance.png)  
![Confusion Matrix](images/confusion_matrix.png)

---

## ⚙️ How to Run Locally

1. **Clone this repository**
```bash
git clone https://github.com/your-username/loan-approval-predictor.git
cd loan-approval-predictor
