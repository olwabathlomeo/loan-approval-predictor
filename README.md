# 🏦 Loan Approval Predictor - Machine Learning App

Predict whether a loan will be **Approved** or **Rejected** using applicant data like income, loan amount, CIBIL score, employment status, and asset value. Built with real-world data, trained using Random Forest, and deployed as a Streamlit web application.

---

## 📊 Project Overview

Loan evaluation is often slow, biased, or based on outdated methods. Our ML-powered app brings speed, accuracy, and fairness to the loan approval process.

- ✔️ Predictive model with over 86% accuracy  
- 💡 Trained on cleaned and encoded real-world loan data  
- 🖥️ Streamlit interface for easy testing and visualization  

---

## 📂 Project Structure

```
loan-approval-predictor/
├── data/
│   └── Loan_approval_dataset.csv
├── notebooks/
│   └── eda_model_building.ipynb
├── app/
│   └── streamlit_app.py
├── models/
│   └── best_rf_model.pkl
├── images/
│   ├── CIBIL_score_boxplot.png
│   ├── feature_importance.png
│   ├── confusion_matrix.png
├── README.md
├── requirements.txt
```

---

## 🧪 Features Used

| Feature          | Description                         |
|------------------|-------------------------------------|
| `income_annum`   | Annual income of the applicant      |
| `loan_amount`    | Requested loan amount               |
| `cibil_score`    | Credit score of the applicant       |
| `education`      | Graduate / Not Graduate             |
| `self_employed`  | Yes / No                            |
| `asset_value`    | Value of applicant's assets         |

---

## 🤖 Model Performance

- **Model Used**: Random Forest Classifier  
- **Accuracy**: `86.3%`

### Evaluation Metrics:
- Precision, Recall, F1-Score  
- Confusion Matrix  
- Feature Importance Plot  

---

## 🌐 App Demo (Streamlit)

The app allows users to input values and receive instant predictions.

**Example Inputs:**
- Annual Income: `550000`
- Loan Amount: `150000`
- CIBIL Score: `725`
- Education: `Graduate`
- Self-Employed: `No`
- Asset Value: `200000`

**Output:**
✅ Loan Approved with 87.5% confidence

---

## 📸 Visualizations

![CIBIL Score Boxplot](images/CIBIL_score_boxplot.png)  
![Feature Importance](images/feature_importance.png)  
![Confusion Matrix](images/confusion_matrix.png)

---

## ⚙️ How to Run Locally

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/loan-approval-predictor.git
   cd loan-approval-predictor
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**  
   ```bash
   streamlit run app/streamlit_app.py
   ```

---

## 👤 Author

**Olwa Bathlomeo Onyangi**  
📧 bathlomeoolwa@gmail.com  
🌐 [GitHub Profile](https://github.com/bathlomeoolwa)

---

## 🛡️ License

This project is licensed under the **MIT License** — feel free to use, modify, and share it.
