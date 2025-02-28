import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model and scaler
model = joblib.load("/Users/lohithkumar/Churn_Telecom/model/churn_model.pkl")
scaler = joblib.load("/Users/lohithkumar/Churn_Telecom/model/scaler.pkl")
column_names = joblib.load("/Users/lohithkumar/Churn_Telecom/model/column_names.pkl")

# Title of the dashboard
st.title("Telecom Customer Churn Prediction Dashboard")

# Input fields for user data
st.sidebar.header("Customer Details")
age = st.sidebar.number_input("Age", min_value=0)
num_dependents = st.sidebar.number_input("Number of Dependents", min_value=0)
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0)
calls_made = st.sidebar.number_input("Calls Made", min_value=0)
sms_sent = st.sidebar.number_input("SMS Sent", min_value=0)
data_used = st.sidebar.number_input("Data Used (GB)", min_value=0.0)
tenure = st.sidebar.number_input("Tenure (months)", min_value=0)
telecom_partner = st.sidebar.selectbox("Telecom Partner", ["Airtel", "Jio", "Vodafone"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
state = st.sidebar.selectbox("State", ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"])
city = st.sidebar.selectbox("City", ["Mumbai", "Bangalore", "Delhi"])

# Convert categorical inputs to numerical
telecom_mapping = {"Airtel": 0, "Jio": 1, "Vodafone": 2}
gender_mapping = {"Male": 0, "Female": 1}
state_mapping = {"Andhra Pradesh": 0, "Arunachal Pradesh": 1, "Assam": 2, "Bihar": 3, "Chhattisgarh": 4, "Goa": 5, "Gujarat": 6, "Haryana": 7, "Himachal Pradesh": 8, "Jharkhand": 9, "Karnataka": 10, "Kerala": 11, "Madhya Pradesh": 12, "Maharashtra": 13, "Manipur": 14, "Meghalaya": 15, "Mizoram": 16, "Nagaland": 17, "Odisha": 18, "Punjab": 19, "Rajasthan": 20, "Sikkim": 21, "Tamil Nadu": 22, "Telangana": 23, "Tripura": 24, "Uttar Pradesh": 25, "Uttarakhand": 26, "West Bengal": 27}
city_mapping = {"Mumbai": 0, "Bangalore": 1, "Delhi": 2}

telecom_encoded = telecom_mapping[telecom_partner]
gender_encoded = gender_mapping[gender]
state_encoded = state_mapping[state]
city_encoded = city_mapping[city]

# Prepare input data as a dictionary
input_data = {
    'age': age,
    'num_dependents': num_dependents,
    'estimated_salary': estimated_salary,
    'calls_made': calls_made,
    'sms_sent': sms_sent,
    'data_used': data_used,
    'tenure': tenure,
    'telecom_partner_Airtel': 1 if telecom_partner == "Airtel" else 0,
    'telecom_partner_Jio': 1 if telecom_partner == "Jio" else 0,
    'telecom_partner_Vodafone': 1 if telecom_partner == "Vodafone" else 0,
    'gender_Male': 1 if gender == "Male" else 0,
    'gender_Female': 1 if gender == "Female" else 0,
    'state_Maharashtra': 1 if state == "Maharashtra" else 0,
    'state_Karnataka': 1 if state == "Karnataka" else 0,
    'state_Delhi': 1 if state == "Delhi" else 0,
    'city_Mumbai': 1 if city == "Mumbai" else 0,
    'city_Bangalore': 1 if city == "Bangalore" else 0,
    'city_Delhi': 1 if city == "Delhi" else 0
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Align columns with the training data
input_df = input_df.reindex(columns=column_names, fill_value=0)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Make prediction
if st.sidebar.button("Predict Churn"):
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[:, 1]
    st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    st.write(f"Churn Probability: {probability[0]:.2f}")
