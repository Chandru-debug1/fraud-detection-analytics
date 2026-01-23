import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="centered"
)

# =====================================
# Load Model Artifacts
# =====================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_artifacts()

# =====================================
# App Header
# =====================================
st.title("💳 Fraud Detection System")
st.markdown(
    """
    This application predicts whether a **financial transaction is fraudulent**
    using a **machine learning model trained on historical transaction data**.
    """
)

st.divider()

# =====================================
# User Input Section
# =====================================
st.subheader("🔢 Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Step", min_value=0, value=1)
    amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
    transaction_type = st.selectbox(
        "Transaction Type",
        ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    )

with col2:
    oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=5000.0)
    newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, value=4000.0)
    oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, value=1000.0)

st.divider()

# =====================================
# Prediction Logic
# =====================================
if st.button("🔍 Predict Fraud", use_container_width=True):

    # Create input dataframe
    input_df = pd.DataFrame([{
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    # Numerical features
    num_features = [
        "step", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest"
    ]

    # Preprocessing
    num_scaled = scaler.transform(input_df[num_features])
    cat_encoded = encoder.transform(input_df[["type"]])

    final_input = np.hstack([num_scaled, cat_encoded])

    # Prediction
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    # Output
    st.divider()

    if prediction == 1:
        st.error("🚨 **Fraudulent Transaction Detected**")
        st.metric(label="Fraud Probability", value=f"{probability:.2%}")
    else:
        st.success("✅ **Transaction is Legitimate**")
        st.metric(label="Fraud Probability", value=f"{probability:.2%}")

# =====================================
# Footer
# =====================================
st.divider()
st.caption(
    "📌 Model trained using Gradient Boosting / Random Forest / Logistic Regression.\n"
    "Recall prioritized to reduce missed fraud cases."
)
