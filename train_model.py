import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# Load the dataset
data = pd.read_csv("/Users/lohithkumar/Churn_Telecom/data/telecom_churn.csv")
print(data.head())
print(data.columns)
#print(data.isnull().sum())



# Preprocessing
data = data.dropna()
label_encoder = LabelEncoder()
data['churn'] = label_encoder.fit_transform(data['churn'])
data = pd.get_dummies(data, drop_first=True)


# Split features and target
X = data.drop(columns=['churn'])
y = data['churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

# Train a AdaBoostClassifier
model_1 =  AdaBoostClassifier(random_state=42)
model_1.fit(X_train, y_train)

# Evaluate the model
y_pred = model_1.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))



# Save the model and scaler
joblib.dump(model, "/Users/lohithkumar/Churn_Telecom/model/churn_model.pkl")
joblib.dump(scaler, "/Users/lohithkumar/Churn_Telecom/model/scaler.pkl")
joblib.dump(X.columns, "/Users/lohithkumar/Churn_Telecom/model/column_names.pkl")