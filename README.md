# Predictive Customer Churn Analysis for Telecom Companies

This project aims to predict customer churn for telecom companies using historical customer data. It includes a machine learning pipeline for preprocessing, feature engineering, model training, and deployment as a REST API. Additionally, a Streamlit dashboard is provided for business stakeholders to interact with the model and gain insights.

---

## **Features**
- **Data Preprocessing**: Handles missing values, encodes categorical variables, and scales numerical features.
- **Feature Engineering**: Creates new features like tenure and one-hot encodes categorical variables.
- **Model Training**: Uses a Random Forest classifier to predict churn.
- **REST API**: Deploys the trained model as a Flask API for real-time predictions.
- **Streamlit Dashboard**: Provides an interactive dashboard for business stakeholders to input customer data and view churn predictions.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Flask, Streamlit, Joblib
- **Deployment**: Flask (API), Streamlit (Dashboard)

---

## **Folder Structure**
telecom-churn-project/
├── data/
│ └── telecom_churn.csv # Dataset for training
├── model/
│ ├── churn_model.pkl # Trained model
│ ├── scaler.pkl # Feature scaler
│ └── column_names.pkl # Column names for one-hot encoding
├── app.py # Flask API
├── dashboard.py # Streamlit dashboard
└── train_model.py # Script for preprocessing and training

Copy

---

## **Setup Instructions**

### **1. Prerequisites**
- Python 3.8 or higher
- Install required libraries:
  ```bash
  pip install pandas numpy scikit-learn xgboost flask streamlit joblib
2. Clone the Repository
bash
Copy
git clone https://github.com/your-username/telecom-churn-project.git
cd telecom-churn-project
3. Train the Model
Place your dataset (telecom_churn.csv) in the data/ folder.

Run the training script:

bash
Copy
python train_model.py
This will:

Preprocess the data.

Train a Random Forest model.

Save the model, scaler, and column names in the model/ folder.

4. Run the Flask API
Start the Flask API:

bash
Copy
python app.py
The API will be available at http://127.0.0.1:5000.

API Endpoint
POST /predict:

Input: JSON with customer details.

Example:

json
Copy
{
  "age": 35,
  "num_dependents": 2,
  "estimated_salary": 50000,
  "calls_made": 100,
  "sms_sent": 50,
  "data_used": 10.5,
  "tenure": 12,
  "telecom_partner": "Jio",
  "gender": "Male",
  "state": "Maharashtra",
  "city": "Mumbai"
}
Output:

json
Copy
{
  "prediction": 0,
  "probability": 0.12
}
5. Run the Streamlit Dashboard
Start the Streamlit dashboard:

bash
Copy
streamlit run dashboard.py
Open the URL provided in the terminal to interact with the dashboard.

Usage
1. Flask API
Use tools like Postman or curl to send POST requests to the API.

Example:

bash
Copy
curl -X POST -H "Content-Type: application/json" -d '{
  "age": 35,
  "num_dependents": 2,
  "estimated_salary": 50000,
  "calls_made": 100,
  "sms_sent": 50,
  "data_used": 10.5,
  "tenure": 12,
  "telecom_partner": "Jio",
  "gender": "Male",
  "state": "Maharashtra",
  "city": "Mumbai"
}' http://127.0.0.1:5000/predict
2. Streamlit Dashboard
Input customer details in the sidebar.

Click Predict Churn to see the prediction and churn probability.

Future Enhancements
Add more machine learning models (e.g., XGBoost, Neural Networks).

Deploy the API and dashboard on the cloud (e.g., AWS, Heroku).

Add real-time data streaming for up-to-date predictions.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or feedback, please contact:

Your Name

Email: your.email@example.com

GitHub: your-username

Copy

---

### **How to Use the README.md File**

1. Save the content above in a file named `README.md` in the root of your project folder.
2. Update the placeholders (e.g., `your-username`, `your.email@example.com`) with your actual details.
3. Push the `README.md` file to your GitHub repository (if applicable).

---

This README file provides a comprehensive guide for anyone looking to understand, set up, and use your project. Let me know if you need further assistance! 🚀
