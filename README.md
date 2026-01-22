# 🚨 Fraud Detection Analytics Platform

An end-to-end fraud detection project combining:
- Machine Learning (Python)
- Interactive Dashboards (Power BI)
- Real-time Prediction App (Streamlit)

---

## 📌 Project Overview
This project analyzes financial transaction data to identify fraudulent behavior.
It combines exploratory data analysis, predictive modeling, and business intelligence
to deliver actionable insights.

---

## 🧠 Machine Learning
- Models used: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Evaluation metrics: Recall, F1-Score, ROC-AUC
- Best model selected based on Recall to minimize missed fraud cases

---

## 📊 Power BI Dashboard
Key insights:
- Fraud concentration by transaction type
- Fraud rate increases with transaction amount
- Financial loss contribution by fraud category

> Power BI dashboard file is available in `/powerbi`  
> Screenshots included for preview.

---

## 🖥️ Streamlit Application
- Real-time fraud prediction
- Model artifacts serialized using joblib
- User-friendly interface for transaction input

---

## 🚀 How to Run Streamlit App
```bash
pip install -r requirements.txt
streamlit run app.py
