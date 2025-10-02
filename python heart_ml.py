import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from collections import Counter

df = pd.read_csv(r"C:\AI-AGENT\heart attack\archive\heart-attack-risk-prediction-dataset.csv")
print("Dataset loaded successfully!\n")

if 'Heart Attack Risk (Text)' in df.columns:
    df = df.drop(columns=['Heart Attack Risk (Text)'])

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

X = df.drop(columns=['Heart Attack Risk (Binary)'])
y = df['Heart Attack Risk (Binary)']

numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='mean')
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
if len(categorical_cols) > 0:
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print("Class distribution after SMOTE:", Counter(y_res))

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("\nXGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

# Interactive input for a new patient with validation
print("\nEnter new patient details:")

new_patient_data = []

for col in X.columns:
    while True:
        value = input(f"{col}: ")
        if col == "Gender":
            if value.lower() in ['male', 'm']:
                new_patient_data.append(1)
                break
            elif value.lower() in ['female', 'f']:
                new_patient_data.append(0)
                break
            else:
                print("Invalid input. Enter 'Male' or 'Female'.")
        else:
            try:
                new_patient_data.append(float(value))
                break
            except ValueError:
                print("Invalid input. Enter a numeric value.")

new_patient_array = np.array([new_patient_data])
new_patient_scaled = scaler.transform(new_patient_array)
prediction = xgb_model.predict(new_patient_scaled)

print("\nPrediction for new patient:")
print("High Risk of Heart Attack" if prediction[0] == 1 else "Low Risk of Heart Attack")
