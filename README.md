# 💓 Heart Attack Risk Prediction using Machine Learning

This project implements a Machine Learning-based system to predict the risk of heart attack (High/Low) using health-related parameters such as age, cholesterol, blood pressure, diabetes, smoking habits, and more.

The solution integrates data preprocessing, class imbalance handling, model training, and real-time prediction, making it suitable for healthcare insights and early risk detection.

 # 🚀 Project Workflow

Data Preprocessing

Missing values imputed using SimpleImputer

Categorical encoding (Gender → Male=1, Female=0)

Feature scaling using StandardScaler

Class Imbalance Handling

Used SMOTE (Synthetic Minority Oversampling Technique) to balance target classes

# Model Training & Evaluation

🌲 Random Forest Classifier

⚡ XGBoost Classifier

# Metrics: Accuracy, Confusion Matrix, Classification Report

# Real-Time Prediction

Users can input health details (age, cholesterol, BP, etc.)

Model outputs High Risk / Low Risk

# 📊 Technologies & Libraries

Python 🐍

Pandas, NumPy

Scikit-learn

XGBoost

Imbalanced-learn (SMOTE)
