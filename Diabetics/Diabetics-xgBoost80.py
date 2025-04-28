import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load the dataset
diabetes_data = pd.read_csv('Diabetics\diabetics.csv')

# Splitting features and target
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X, Y = smote.fit_resample(X, Y)

# Splitting data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Initialize the XGBoost Classifier
xgboost_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6
)

# Train the model
xgboost_model.fit(X_train, Y_train)

# Evaluate the model on test data
xgboost_preds = xgboost_model.predict(X_test)
xgboost_accuracy = accuracy_score(Y_test, xgboost_preds)

# Metrics: Accuracy, Precision, Recall, F1-Score
print(f"XGBoost Test Accuracy: {xgboost_accuracy:.4f}")
print("\nXGBoost Classification Report on Test Data:")
print(classification_report(Y_test, xgboost_preds))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, xgboost_preds))

