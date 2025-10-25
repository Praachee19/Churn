import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_model.pkl")

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")

# Inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 10.0, 120.0, 50.0)
contract_type = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])

# Map input
contract_map = {
    'Month-to-month': [0, 0],
    'One year': [1, 0],
    'Two year': [0, 1]
}

features = pd.DataFrame([[
    tenure,
    monthly_charges,
    *contract_map[contract_type]
]], columns=['tenure', 'MonthlyCharges', 'Contract_One year', 'Contract_Two year'])

# Predict
if st.button("Predict Churn"):
    pred = model.predict_proba(features)[:, 1][0]
    st.success(f"Churn Probability: {pred:.2%}")
