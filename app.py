from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("/Users/lohithkumar/Churn_Telecom/model/churn_model.pkl")
scaler = joblib.load("/Users/lohithkumar/Churn_Telecom/model/scaler.pkl")
column_names = joblib.load("/Users/lohithkumar/Churn_Telecom/model/column_names.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Prepare input data as a dictionary
    input_data = {
        'age': data['age'],
        'num_dependents': data['num_dependents'],
        'estimated_salary': data['estimated_salary'],
        'calls_made': data['calls_made'],
        'sms_sent': data['sms_sent'],
        'data_used': data['data_used'],
        'tenure': data['tenure'],
        'telecom_partner_Airtel': 1 if data['telecom_partner'] == "Airtel" else 0,
        'telecom_partner_Jio': 1 if data['telecom_partner'] == "Jio" else 0,
        'telecom_partner_Vodafone': 1 if data['telecom_partner'] == "Vodafone" else 0,
        'gender_Male': 1 if data['gender'] == "Male" else 0,
        'gender_Female': 1 if data['gender'] == "Female" else 0,
        'state_Maharashtra': 1 if data['state'] == "Maharashtra" else 0,
        'state_Karnataka': 1 if data['state'] == "Karnataka" else 0,
        'state_Delhi': 1 if data['state'] == "Delhi" else 0,
        'city_Mumbai': 1 if data['city'] == "Mumbai" else 0,
        'city_Bangalore': 1 if data['city'] == "Bangalore" else 0,
        'city_Delhi': 1 if data['city'] == "Delhi" else 0
    }
    
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Align columns with the training data
    input_df = input_df.reindex(columns=column_names, fill_value=0)
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[:, 1]
    
    return jsonify({
        'prediction': int(prediction[0]),
        'probability': float(probability[0])
    })

if __name__ == '__main__':
    app.run(debug=True)