# Predictive Customer Churn Analysis for Telecom Companies

An end-to-end machine learning project that predicts which telecom customers are likely to churn. It covers preprocessing, feature engineering, model training, a Flask REST API for real-time predictions, and a Streamlit dashboard for business users.

## Features

- **Data preprocessing:** handles missing values, encodes categorical variables and scales numerical features.
- **Feature engineering:** creates features such as tenure and one-hot encodes categorical variables.
- **Model training:** a Random Forest classifier predicts churn.
- **REST API:** the trained model is served as a Flask API (`app.py`).
- **Streamlit dashboard:** enter customer details and see the predicted churn and its probability (`dashboard.py`).

## Technologies

- Python
- Pandas, NumPy, scikit-learn, XGBoost, Joblib
- Flask (API) and Streamlit (dashboard)

## Project structure

```
Churn_Data_Prediction/
├── app.py            # Flask API
├── dashboard.py      # Streamlit dashboard
└── train_model.py    # Preprocessing and model training
```

`train_model.py` expects the dataset at `data/telecom_churn.csv` and saves the trained model, scaler and column names to a `model/` folder. The dataset is not included in this repository, so add your own copy before training.

## Setup

1. Install Python 3.8 or higher and the dependencies:

   ```bash
   pip install pandas numpy scikit-learn xgboost flask streamlit joblib
   ```

2. Clone the repository:

   ```bash
   git clone https://github.com/SaiLohithKumar/Churn_Data_Prediction.git
   cd Churn_Data_Prediction
   ```

3. Place `telecom_churn.csv` in a `data/` folder, then train the model:

   ```bash
   python train_model.py
   ```

4. Start the Flask API (runs at http://127.0.0.1:5000):

   ```bash
   python app.py
   ```

5. Start the dashboard:

   ```bash
   streamlit run dashboard.py
   ```

## API

`POST /predict` takes JSON with the customer details and returns the prediction and churn probability.

Example request:

```json
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
```

Example response:

```json
{ "prediction": 0, "probability": 0.12 }
```

Using curl:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"age": 35, "num_dependents": 2, "estimated_salary": 50000, "calls_made": 100, "sms_sent": 50, "data_used": 10.5, "tenure": 12, "telecom_partner": "Jio", "gender": "Male", "state": "Maharashtra", "city": "Mumbai"}' \
  http://127.0.0.1:5000/predict
```

## Future improvements

- Compare more models (XGBoost, neural networks).
- Deploy the API and dashboard to the cloud.
- Add real-time data streaming for up-to-date predictions.

## Author

Sai Lohith Kumar Paidi - [GitHub](https://github.com/SaiLohithKumar) · [LinkedIn](https://www.linkedin.com/in/paidi-sai-lohith-kumar-762407240/)
